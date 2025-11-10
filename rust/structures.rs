use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::Deref;
use std::ops::DerefMut;

use serde::ser::SerializeMap;
use serde::ser::SerializeTuple;
use serde::{Serialize, Serializer};

/// u32 allows up to 4,294,967,296 qubits
/// If we used u16 it would only allow 65,536 qubits
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, Serialize)]
pub struct Qubit(u32);
impl Qubit {
    pub fn new<Q: Into<Qubit>>(q: Q) -> Self {
        q.into()
    }

    pub fn get_index(&self) -> u32 {
        return self.0;
    }
}
impl From<u32> for Qubit {
    fn from(n: u32) -> Self {
        Qubit(n)
    }
}
impl Display for Qubit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type Location = u32;
pub type ID = u32;

/// Enum for gates in the circuit.
///
/// This is probably the fist place we should look to if we need to lower memory usage.
/// Technically enums have some overhead, and there are other packing techniques we can do.
///
/// repr(u8) forces a small discriminant, u8 is small enough for our operations
#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Operation {
    T,
    TDG,
    CX,
}
impl Serialize for Operation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Operation::T => serializer.serialize_str("t"),
            Operation::TDG => serializer.serialize_str("tdg"),
            Operation::CX => serializer.serialize_str("cx"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Gate {
    op: Operation,
    id: u32,
    q0: Qubit,
    q1: Option<Qubit>,
}

impl Gate {
    pub fn t<T: Into<Qubit>>(id: u32, q: T) -> Self {
        Gate {
            op: Operation::T,
            id,
            q0: q.into(),
            q1: None,
        }
    }

    pub fn tdg<T: Into<Qubit>>(id: u32, q: T) -> Self {
        Gate {
            op: Operation::TDG,
            id,
            q0: q.into(),
            q1: None,
        }
    }

    pub fn cx<T: Into<Qubit>, U: Into<Qubit>>(id: u32, q0: T, q1: U) -> Self {
        Gate {
            op: Operation::CX,
            id,
            q0: q0.into(),
            q1: Some(q1.into()),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn set_id(&mut self, new_id: u32) {
        self.id = new_id;
    }

    pub fn is_double_qubit(&self) -> bool {
        matches!(self.op, Operation::CX)
    }

    pub fn get_qubits(&self) -> (Qubit, Option<Qubit>) {
        (self.q0, self.q1)
    }
}

impl Serialize for Gate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self.q1 {
            Some(q1) => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(&self.q0)?;
                tuple.serialize_element(&q1)?;
                tuple.end()
            }
            None => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(&self.q0)?;
                tuple.end()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Route {
    pub gate: Gate,
    pub path: Vec<Location>,
}

impl Route {
    pub fn new(gate: &Gate, path: Vec<Location>) -> Self {
        Self {
            gate: gate.clone(),
            path: path,
        }
    }
}
impl Serialize for Route {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(4))?;
        state.serialize_entry("id", &self.gate.id)?;
        state.serialize_entry("op", &self.gate.op)?;
        state.serialize_entry("qubits", &self.gate)?;
        state.serialize_entry("path", &self.path)?;
        state.end()
    }
}

#[derive(Debug)]
pub struct Architecture {
    pub height: u32,
    pub width: u32,
    pub qubit_locations: Vec<Location>,
    pub magic_locations: Vec<Location>,
}
impl Architecture {
    pub fn square_sparse(qubits: usize, magic_faces: bool) -> Self {
        let grid_width =
            2 * ((qubits as f32).sqrt().ceil() as u32) + 1 + (if magic_faces { 2 } else { 0 });
        let grid_height = grid_width;
        let mut qlocs: Vec<u32> = Vec::new();
        // Edge lists
        let mut top = Vec::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut bottom = Vec::new();
        for i in 0..grid_height {
            for j in 0..grid_width {
                let loc = i * grid_width + j;
                // Top
                if i == 0 {
                    top.push(loc);
                    continue;
                }
                // Bottom
                if i == grid_height - 1 {
                    bottom.push(loc);
                    continue;
                }
                // Left
                if j == 0 {
                    left.push(loc);
                    continue;
                }
                // Right
                if j == grid_width - 1 {
                    right.push(loc);
                    continue;
                }
                if i % 2 == 0 && j % 2 == 0 {
                    qlocs.push(loc);
                }
            }
        }

        let mut edge_locations: Vec<u32> = Vec::new();
        edge_locations.extend(top);
        edge_locations.extend(right);
        edge_locations.extend(bottom.into_iter().rev());
        edge_locations.extend(left);

        let mut seen = HashSet::new();
        let mut unique: Vec<u32> = Vec::new();
        for x in edge_locations {
            if seen.insert(x) {
                unique.push(x);
            }
        }

        let mut msf: Vec<u32> = Vec::new();
        for i in (1..unique.len()).step_by(2) {
            msf.push(unique[i]);
        }

        Architecture {
            height: grid_height,
            width: grid_width,
            qubit_locations: qlocs,
            magic_locations: msf,
        }
    }
}
impl Serialize for Architecture {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_map(Some(4))?;
        state.serialize_entry("height", &self.height)?;
        state.serialize_entry("width", &self.width)?;
        state.serialize_entry("alg_qubits", &self.qubit_locations)?;
        state.serialize_entry("magic_states", &self.magic_locations)?;
        state.end()
    }
}

#[derive(Clone, Debug)]
pub struct Circuit {
    pub gates: Vec<Gate>,
    pub qubits: HashSet<Qubit>,
}
impl Circuit {
    pub fn new() -> Self {
        Circuit {
            gates: Vec::new(),
            qubits: HashSet::new(),
        }
    }
    pub fn add_gate(&mut self, gate: Gate) {
        let (q0, maybe_q1) = gate.get_qubits();
        if !self.qubits.contains(&q0) {
            self.qubits.insert(q0);
        }
        if let Some(q1) = maybe_q1 {
            if !self.qubits.contains(&q1) {
                self.qubits.insert(q1);
            }
        }
        self.gates.push(gate);
    }
    pub fn split(&self, chunk_size: usize) -> Vec<Circuit> {
        let mut chunks: Vec<Circuit> = Vec::new();
        let mut gates: Vec<Gate> = Vec::new();
        let mut i: usize = 0;
        for gate in self.gates.iter() {
            if i == chunk_size {
                i = 0;
                chunks.push(Circuit {
                    gates: gates.clone(),
                    qubits: self.qubits.clone(),
                });
                gates.clear();
            }
            gates.push(gate.clone());
            i += 1;
        }
        chunks.push(Circuit {
            gates: gates.clone(),
            qubits: self.qubits.clone(),
        });
        chunks
    }
    pub fn sort_gates(&mut self) {
        let mut layers: Vec<Vec<Gate>> = Vec::new();
        let mut qubit_depths: HashMap<Qubit, usize> = HashMap::new();
        for gate in self.gates.iter() {
            let (q0, maybe_q1) = gate.get_qubits();
            let mut max_depth: usize = 0;
            if !qubit_depths.contains_key(&q0) {
                qubit_depths.insert(q0, 0);
            }
            max_depth = max_depth.max(qubit_depths[&q0]);
            if let Some(q1) = maybe_q1 {
                if !qubit_depths.contains_key(&q1) {
                    qubit_depths.insert(q1, 0);
                }
                max_depth = max_depth.max(qubit_depths[&q1]);
                qubit_depths.insert(q1, max_depth + 1);
            }
            qubit_depths.insert(q0, max_depth + 1);
            if max_depth == layers.len() {
                layers.push(Vec::new());
            }
            layers[max_depth].push(gate.clone());
        }
        self.gates = layers
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(i, mut gate)| {
                gate.id = i as u32;
                gate
            })
            .collect();
    }
}
impl Serialize for Circuit {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.gates.serialize(serializer)
    }
}

#[derive(Clone)]
pub struct QubitMap(HashMap<Qubit, Location>);
impl QubitMap {
    pub fn new() -> Self {
        return QubitMap(HashMap::new());
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Qubit, &Location)> {
        self.0.iter()
    }
}
impl Deref for QubitMap {
    type Target = HashMap<Qubit, Location>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for QubitMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Serialize for QubitMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut ser_map = serializer.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            ser_map.serialize_entry(&k.to_string(), v)?;
        }
        ser_map.end()
    }
}

pub type RoutingLayer = Vec<Route>;
#[derive(Serialize, Clone)]
pub struct GateRouting(Vec<RoutingLayer>);
impl GateRouting {
    pub fn new() -> Self {
        GateRouting(Vec::new())
    }
    pub fn merge_routings(routings: Vec<GateRouting>) -> Self {
        GateRouting(routings.into_iter().flat_map(|r| r.0).collect())
    }
}
impl IntoIterator for GateRouting {
    type Item = RoutingLayer;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl Deref for GateRouting {
    type Target = Vec<RoutingLayer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for GateRouting {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Serialize)]
pub struct MappingAndRouting {
    pub map: QubitMap,
    pub steps: GateRouting,
    pub arch: Architecture,
    pub gates: Circuit,
}