use crate::structures::{
    Architecture, Circuit, Gate, GateRouting, Location, Qubit, QubitMap, Route,
};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cmp::{Ordering, max};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::u32;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: u32,
    position: u32,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// ChatGPT astar grid search
fn astar_grid(grid: &[Vec<u32>], start: u32, goal: u32) -> Vec<u32> {
    let h = grid.len() as u32;
    let w = grid[0].len() as u32;
    let mut g_score = vec![u32::MAX; (h * w) as usize];
    let mut came_from = vec![u32::MAX; (h * w) as usize];

    let heuristic = |v: u32| -> u32 {
        let x = v % w;
        let y = v / w;
        let gx = goal % w;
        let gy = goal / w;
        x.abs_diff(gx) + y.abs_diff(gy)
    };

    let mut open_set = BinaryHeap::new();
    open_set.push(State {
        cost: heuristic(start),
        position: start,
    });
    g_score[start as usize] = 0;

    while let Some(State {
        position: current, ..
    }) = open_set.pop()
    {
        if current == goal {
            let mut path = Vec::new();
            let mut v = goal;
            loop {
                path.push(v);
                let prev = came_from[v as usize];
                if prev == u32::MAX {
                    break;
                }
                v = prev;
            }
            path.reverse();
            return path;
        }

        let cx = current % w;
        let cy = current / w;
        let neighbors = [
            (cx.wrapping_add(1), cy),
            (cx.wrapping_sub(1), cy),
            (cx, cy.wrapping_add(1)),
            (cx, cy.wrapping_sub(1)),
        ];

        for (nx, ny) in neighbors {
            if nx >= w || ny >= h {
                continue;
            }
            if grid[ny as usize][nx as usize] == u32::MAX {
                continue;
            }

            let neighbor = ny * w + nx;
            let tentative_g = g_score[current as usize].saturating_add(1);
            if tentative_g < g_score[neighbor as usize] {
                came_from[neighbor as usize] = current;
                g_score[neighbor as usize] = tentative_g;
                open_set.push(State {
                    cost: tentative_g + heuristic(neighbor),
                    position: neighbor,
                });
            }
        }
    }

    Vec::new()
}

fn vertical_neighbors(loc: Location, width: u32, height: u32) -> Vec<Location> {
    let mut neighbors = Vec::new();
    let down = loc + width;
    let up = loc - width;
    if loc / width != 0 {
        neighbors.push(up);
    }
    if loc / width != height - 1 {
        neighbors.push(down);
    }
    neighbors
}

fn horizontal_neighbors(loc: Location, width: u32) -> Vec<Location> {
    let mut neighbors = Vec::new();
    let left = loc - 1;
    let right = loc + 1;
    if loc % width != 0 {
        neighbors.push(left);
    }
    if loc % width != width - 1 {
        neighbors.push(right);
    }
    neighbors
}

/// Function for routing a gate
/// Routes a gate, returns the route, modifies the to_remove
fn route_gate(
    gate: &Gate,
    mapping: &QubitMap,
    arch: &Architecture,
    to_remove: &mut HashSet<u32>,
) -> Option<Route> {
    // Build occupancy grid: 0 = free, u32::MAX = blocked
    let mut grid = vec![vec![0; arch.width as usize]; arch.height as usize];
    for &loc in to_remove.iter() {
        let y = (loc / arch.width) as usize;
        let x = (loc % arch.width) as usize;
        grid[y][x] = u32::MAX;
    }

    let mut shortest_path_len = u32::MAX;
    let mut shortest_path = Vec::new();
    let mut pairs = Vec::new();

    let (q0, maybe_q1) = gate.get_qubits();
    if let Some(q1) = maybe_q1 {
        // Source q0 vertical neighbors, destination q1 horizontal neighbors
        for vn in vertical_neighbors(*mapping.get(&q0)?, arch.width, arch.height) {
            for hn in horizontal_neighbors(*mapping.get(&q1)?, arch.width) {
                pairs.push((vn, hn));
            }
        }
    } else {
        // Source q0 vertical neighbors, destination = any magic state horizontal neighbor
        let target = mapping.get(&q0)?;
        let mut sorted_msf = arch.magic_locations.clone();

        // Sort magic state faces by Manhattan distance to q0 face
        sorted_msf.sort_by(|&m1, &m2| {
            let manhattan = |a: u32| -> u32 {
                let ax = a % arch.width;
                let ay = a / arch.width;
                let tx = target % arch.width;
                let ty = target / arch.width;
                ax.abs_diff(tx) + ay.abs_diff(ty)
            };
            manhattan(m1).cmp(&manhattan(m2))
        });

        for magic_state in sorted_msf {
            for vn in vertical_neighbors(*mapping.get(&q0)?, arch.width, arch.height) {
                for hn in horizontal_neighbors(magic_state, arch.width) {
                    pairs.push((vn, hn));
                }
            }
        }
    }

    // Filter out pairs which overlap with to_remove
    // println!("{}",pairs.len());
    pairs.retain(|(s, t)| !to_remove.contains(s) && !to_remove.contains(t));

    for (s, t) in pairs {
        let path = astar_grid(&grid, s, t);
        if !path.is_empty() {
            let distance = path.len() as u32;
            if distance < shortest_path_len {
                shortest_path_len = distance;
                shortest_path = path;
            }
        }
    }

    if shortest_path.is_empty() {
        return None; // no valid route
    }

    // Update occupancy set
    for &loc in &shortest_path {
        to_remove.insert(loc);
    }

    Some(Route::new(&gate, shortest_path))
}

fn initialize_to_remove(map: &QubitMap, arch: &Architecture) -> HashSet<Location> {
    let mut to_remove = HashSet::new();
    for (_, &loc) in map.iter() {
        to_remove.insert(loc);
    }
    for &loc in arch.magic_locations.iter() {
        to_remove.insert(loc);
    }
    to_remove
}

fn try_order(
    order: &[u32],
    executable_gates: &BTreeMap<u32, &Gate>,
    map: &QubitMap,
    arch: &Architecture,
) -> Vec<Route> {
    let mut step = Vec::new();
    let mut to_remove = initialize_to_remove(map, arch);

    for &id in order {
        if let Some(&gate) = executable_gates.get(&id) {
            // Route the gate, mutates to_remove
            if let Some(route) = route_gate(gate, map, arch, &mut to_remove) {
                step.push(route);
            }
        }
    }
    step
}

fn criticality_fast(step: &[Route], gate_crits: &HashMap<u32, u32>) -> i32 {
    step.iter()
        .map(|r| *gate_crits.get(&r.gate.id()).unwrap() as i32)
        .sum()
}

/// Differences from dascot:
/// I do not try to swap cnots and t gates, because im not sure dascot even does
/// that! Since the order is randomized, the distinction between the two does not
/// change.
/// I am only implementing the criticality heuristic.
fn best_realizable_set_found(
    _id_gates: &BTreeMap<u32, &Gate>,
    executable_gates: &BTreeMap<u32, &Gate>,
    mapping: &QubitMap,
    arch: &Architecture,
    gate_crits: &HashMap<u32, u32>,
    mut temperature: f64,
    cooling_rate: f64,
    termination_temp: f64,
) -> (Vec<Route>, i32) {
    let mut single_ids = Vec::new();
    let mut double_ids = Vec::new();
    let mut best_order = Vec::new();

    for (&id, gate) in executable_gates {
        if gate.is_double_qubit() {
            double_ids.push(id);
        } else {
            single_ids.push(id);
        }
        best_order.push(id);
    }

    // Default best order is random
    let mut rng = thread_rng();
    best_order.shuffle(&mut rng);
    let mut best_step = try_order(&best_order, executable_gates, mapping, arch);
    let mut current_order = best_order.clone();
    let mut current_step = best_step.clone();
    let mut orders_tried = 1;

    // Trivial case
    if executable_gates.len() < 2 {
        return (best_step, 1);
    }

    // Brute force case
    if double_ids.len() < 5 && single_ids.len() < 5 && cooling_rate != 1.0 {
        // Not implementing for now
    }

    // Random number generator for swaps
    let dis = rand::distributions::Uniform::new(0, executable_gates.len());

    // Simulated Annealing
    while temperature > termination_temp {
        let mut new_order = current_order.clone();

        // Swap two gates in the order
        let index1 = dis.sample(&mut rng);
        let mut index2;
        loop {
            index2 = dis.sample(&mut rng);
            if index2 != index1 {
                break;
            }
        }
        new_order.swap(index1, index2);

        let new_step = try_order(&new_order, executable_gates, mapping, arch);
        orders_tried += 1;

        let delta_curr =
            criticality_fast(&current_step, gate_crits) - criticality_fast(&new_step, gate_crits);
        let delta_best =
            criticality_fast(&best_step, gate_crits) - criticality_fast(&new_step, gate_crits);

        if delta_curr < 0 || rng.r#gen::<f64>() < (-delta_curr as f64 / temperature).exp() {
            current_order = new_order.clone();
            current_step = new_step.clone();
        }

        if delta_best < 0 {
            best_step = new_step;
        }

        // cool down
        temperature *= 1.0 - cooling_rate;
    }

    (best_step, orders_tried)
}

fn executable_subset<'a>(
    id_gates: &BTreeMap<u32, &'a Gate>,
) -> (BTreeMap<u32, &'a Gate>, BTreeMap<u32, &'a Gate>) {
    let mut executable = BTreeMap::new();
    let mut remaining = BTreeMap::new();
    let mut blocked_qubits = HashSet::new();

    for (&id, &gate) in id_gates {
        let (q0, maybe_q1) = gate.get_qubits();
        let mut not_blocked = !blocked_qubits.contains(&q0);
        if let Some(q1) = maybe_q1 {
            not_blocked &= !blocked_qubits.contains(&q1);
        }

        if not_blocked {
            executable.insert(id, gate);
        } else {
            remaining.insert(id, gate);
        }

        blocked_qubits.insert(q0);
        if let Some(q1) = maybe_q1 {
            blocked_qubits.insert(q1);
        }
    }

    (executable, remaining)
}

fn find_gate_crits(gates: &[Gate]) -> HashMap<u32, u32> {
    let mut gate_crits = HashMap::new();
    let mut last_id_per_qubit: HashMap<Qubit, u32> = HashMap::new();

    for gate in gates.iter().rev() {
        let mut max_crit = 1;
        let (q0, maybe_q1) = gate.get_qubits();
        // Update max_crit
        match last_id_per_qubit.get(&q0) {
            Some(id) => max_crit = max(max_crit, *gate_crits.get(id).unwrap()),
            None => (),
        }
        if let Some(q1) = maybe_q1 {
            match last_id_per_qubit.get(&q1) {
                Some(id) => max_crit = max(max_crit, *gate_crits.get(id).unwrap()),
                None => (),
            }
        }
        gate_crits.insert(gate.id(), max_crit);
        // Update gates
        last_id_per_qubit.insert(q0, gate.id());
        if let Some(q1) = maybe_q1 {
            last_id_per_qubit.insert(q1, gate.id());
        }
    }

    gate_crits
}

pub fn sim_anneal_route(
    mapping: &QubitMap,
    arch: &Architecture,
    circ: &Circuit,
    temperature: f64,
    cooling_rate: f64,
    termination_temp: f64,
) -> GateRouting {
    let mut time_steps = GateRouting::new();
    let gate_crits = find_gate_crits(&circ.gates);
    let mut id_gates: BTreeMap<u32, &Gate> = BTreeMap::new();
    circ.gates.iter().for_each(|gate| {
        id_gates.insert(gate.id(), &gate);
    });

    while !id_gates.is_empty() {
        let (executable_gates, remaining_gates) = executable_subset(&id_gates);
        let (step, _) = best_realizable_set_found(
            &id_gates,
            &executable_gates,
            mapping,
            arch,
            &gate_crits,
            temperature,
            cooling_rate,
            termination_temp,
        );

        let routed_ids: HashSet<u32> = step.iter().map(|r| r.gate.id()).collect();
        time_steps.push(step);

        let mut new_id_gates = BTreeMap::new();
        for (id, gate) in remaining_gates {
            new_id_gates.insert(id, gate);
        }
        for (id, gate) in executable_gates {
            if !routed_ids.contains(&id) {
                new_id_gates.insert(id, gate);
            }
        }

        id_gates = new_id_gates;
    }

    time_steps
}

fn refine_route_edges(
    mapping: &QubitMap,
    arch: &Architecture,
    temperature: f64,
    cooling_rate: f64,
    termination_temp: f64,
    routings: Vec<GateRouting>,
    merge_layers: usize,
) -> Vec<GateRouting> {
    let routings_length = routings.len();
    if routings_length <= 1 {
        return routings;
    }
    let mut shortened_routings: Vec<GateRouting> = Vec::with_capacity(routings_length);
    let mut circuits: Vec<Circuit> = vec![Circuit::new(); routings_length - 1];
    for (i, mut routing) in routings.into_iter().enumerate() {
        let adjusted_layers = merge_layers.min(routing.len() / 2);
        let split_idx = routing.len().saturating_sub(adjusted_layers);
        if i != routings_length - 1 {
            let tail = routing.split_off(split_idx);
            let routes: Vec<Route> = tail.into_iter().flatten().collect();
            routes
                .into_iter()
                .for_each(|route| circuits[i].add_gate(route.gate));
        }
        if i != 0 {
            let head = routing.drain(..adjusted_layers);
            let routes: Vec<Route> = head.into_iter().flatten().collect();
            routes
                .into_iter()
                .for_each(|route| circuits[i - 1].add_gate(route.gate));
        }

        shortened_routings.push(routing);
    }
    let merged_routes: Vec<GateRouting> = circuits
        .into_par_iter()
        .map(|c| {
            sim_anneal_route(
                mapping,
                arch,
                &c,
                temperature,
                cooling_rate,
                termination_temp,
            )
        })
        .collect();
    let mut new_routings: Vec<GateRouting> = Vec::with_capacity(routings_length * 2 - 1);
    let mut shortened_routings_iter = shortened_routings.into_iter();
    for routing in merged_routes {
        if let Some(shortened) = shortened_routings_iter.next() {
            new_routings.push(shortened);
        }
        new_routings.push(routing);
    }
    if let Some(last) = shortened_routings_iter.next() {
        new_routings.push(last);
    }
    // new_routings
    //     .iter()
    //     .for_each(|routing| println!("{}", routing.len()));
    // println!("{}", new_routings.len());

    new_routings
}

#[must_use]
pub fn naive_route(mapping: &QubitMap, arch: &Architecture, circ: &Circuit) -> Circuit {
    let routes = splitting_parallel_routing(mapping, arch, circ, 10., 0.1, 0.1, 100, 10);
    let mut new_circ = Circuit::new();
    for route in routes.into_iter().flatten() {
        new_circ.add_gate(route.gate);
    }
    new_circ
}

pub fn splitting_parallel_routing(
    mapping: &QubitMap,
    arch: &Architecture,
    circ: &Circuit,
    temperature: f64,
    cooling_rate: f64,
    termination_temp: f64,
    chunk_size: usize,
    merge_layers: usize,
) -> GateRouting {
    let circ_chunks = circ.split(chunk_size);
    let routings: Vec<GateRouting> = circ_chunks
        // .into_par_iter()
        .into_iter()
        .map(|c| {
            sim_anneal_route(
                mapping,
                arch,
                &c,
                temperature,
                cooling_rate,
                termination_temp,
            )
        })
        .collect();
    // println!("{}", routes.len());
    let routings = refine_route_edges(
        mapping,
        arch,
        temperature,
        cooling_rate,
        termination_temp,
        routings,
        merge_layers,
    );
    // routings
    //     .iter()
    //     .for_each(|routing| println!("{}", routing.len()));

    GateRouting::merge_routings(routings)
}