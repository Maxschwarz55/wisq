use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::routing::{sim_anneal_route, splitting_parallel_routing};
use crate::structures::{Architecture, Circuit, Location, MappingAndRouting, Qubit, QubitMap};
use std::collections::HashSet;
use std::time::Instant;

fn random_neighbor(map: &QubitMap, arch: &Architecture) -> QubitMap {
    let mut rng = thread_rng();

    let qubits: Vec<Qubit> = map.keys().copied().collect();
    let rand_qubit = *qubits
        .choose(&mut rng)
        .expect("QubitMap should not be empty");

    let remaining_qubits: Vec<Qubit> = qubits.into_iter().filter(|q| *q != rand_qubit).collect();

    let mapped_locations: HashSet<Location> = map.values().copied().collect();
    let unmapped_locations: Vec<Location> = arch
        .qubit_locations
        .iter()
        .filter(|&loc| !mapped_locations.contains(loc))
        .copied()
        .collect();

    let total_choices = unmapped_locations.len() + remaining_qubits.len();

    //If no valid neighbors exist, return the map unchanged
    if total_choices == 0 {
        return map.clone();
    }

    let rand_index = rng.gen_range(0..total_choices);

    let mut new_map = map.clone();

    if rand_index < unmapped_locations.len() {
        let new_location = unmapped_locations[rand_index];
        new_map.insert(rand_qubit, new_location);
    } else {
        let qubit_two_index = rand_index - unmapped_locations.len();
        let qubit_two = remaining_qubits[qubit_two_index];

        let location_one = *map.get(&rand_qubit).expect("Should be mapped Qubit");
        let location_two = *map.get(&qubit_two).expect("Should be mapped Qubit");

        new_map.insert(rand_qubit, location_two);
        new_map.insert(qubit_two, location_one);
    }

    new_map
}

pub fn random_map(arch: &Architecture, circ: &Circuit) -> QubitMap {
    let mut result = QubitMap::new();

    // Convert HashMap to Vec for shuffling
    let mut qubits: Vec<Qubit> = circ.qubits.iter().cloned().collect();
    let mut locations = arch.qubit_locations.clone();

    let mut rng = thread_rng();

    locations.shuffle(&mut rng);
    qubits.shuffle(&mut rng);

    //Insert random qubits at random locations
    for (q, l) in qubits.iter().zip(locations.iter()) {
        result.insert(*q, *l);
    }

    result
}

pub fn run_simulated_annealing(
    arch: Architecture,
    circ: &Circuit,
    init_mapping_temp: f64,
    init_routing_temp: f64,
    term_mapping_temp: f64,
    term_routing_temp: f64,
    cooling_mapping_rate: f64,
    cooling_routing_rate: f64,
    deadline: Instant
) -> MappingAndRouting {
    let mut curr_map = random_map(&arch, circ);
    let mut curr_mapping_temp = init_mapping_temp;
    let mut curr_routing_temp = init_routing_temp;


    let curr_res = splitting_parallel_routing(
        &curr_map,
        &arch,
        circ,
        curr_routing_temp,
        cooling_routing_rate,
        term_routing_temp,
        100,
        10,
    );
    let mut best_route = curr_res.clone();
    let mut best_map = curr_map.clone();
    let mut curr_len = curr_res.len();
    let mut min_len = curr_len;

    while curr_mapping_temp > term_mapping_temp && Instant::now() < deadline {
        let next_map = random_neighbor(&curr_map, &arch);
        let next_res = sim_anneal_route(
            &next_map,
            &arch,
            circ,
            curr_routing_temp,
            cooling_routing_rate,
            term_routing_temp,
        );
        let next_len = next_res.len();

        let delta_curr = (next_len as i32) - (curr_len as i32);

        let accept = rand::random::<f64>() < (-(delta_curr as f64) / curr_mapping_temp).exp();

        if next_len < min_len {
            best_route = next_res.clone();
            best_map = next_map.clone();
            min_len = next_len;
            curr_map = next_map;
            curr_len = next_len;
        } else if accept {
            curr_map = next_map;
            curr_len = next_len;
        }

        curr_mapping_temp *= cooling_mapping_rate;
        curr_routing_temp *= cooling_routing_rate;
    }

    MappingAndRouting {
        arch,
        gates: circ.clone(),
        map: best_map,
        steps: best_route,
    }
}