use std::collections::{HashMap, HashSet};
use thiserror::Error;

use crate::structures::*;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Mapping Error: {0}")]
    MappingError(#[from] MappingError),
    #[error("Routing Error: {0}")]
    RoutingError(#[from] RoutingError),
}

#[derive(Debug, Error)]
pub enum MappingError {
    #[error("Missing Qubits")]
    MissingQubits,
    #[error("Invalid Locations")]
    InvalidLocations,
    #[error("Overlapping Locations")]
    OverlappingLocations,
}

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("Unequal Gate Depths, at route {0}")]
    UnequalGateDepths(u32),
    #[error("Invalid Terminal Path Locations, at route {0}")]
    InvalidTerminalPathLocations(u32),
    #[error("Overlapping Path Locations, at route {0}")]
    OverlappingPathLocations(u32),
    #[error("Invalid Path Locations, at route {0}")]
    InvalidPathLocations(u32),
}
impl MappingAndRouting {
    /// # Checks
    /// ## Mapping
    /// * Every qubit in the circuit is accounted for
    /// * Every qubit has a valid location in the circuit's qubit locations
    /// * No qubit has an overlapping location
    /// ## Routing
    /// ### Per Route
    /// * Ensures the depth of the route's gate is equal to the
    /// depth of the gate in the circuit
    /// ### Per Path
    /// * Start and end locations of the path are valid
    /// * No path in a layer have overlapping locations
    /// * Every location in a path is in bounds and doesn't overlap with a magic
    /// location or used qubit location
    pub fn validate(&self) -> Result<(), ValidationError> {
        let qubit_locations: HashSet<&Location> = self.arch.qubit_locations.iter().collect();
        let magic_locations: HashSet<&Location> = self.arch.magic_locations.iter().collect();

        // Validate Mapping
        // -------------------------------
        // Every qubit acounted for
        if self.map.len() != self.gates.qubits.len() {
            Err(MappingError::MissingQubits)?;
        }
        // Every qubit has a valid location and no overlapping locations
        let mut used_qubit_locations: HashSet<Location> = HashSet::new();
        self.map
            .iter()
            .try_for_each(|(_, loc)| -> Result<(), ValidationError> {
                if !qubit_locations.contains(loc) {
                    Err(MappingError::InvalidLocations)?;
                }
                if used_qubit_locations.contains(loc) {
                    Err(MappingError::OverlappingLocations)?;
                }
                used_qubit_locations.insert(*loc);
                Ok(())
            })?;

        // Create depth vec
        // println!("{}", self.gates.gates.len());
        // self.gates.gates.iter().for_each(|g| println!("{}", g.id()));
        let mut gate_depths: Vec<usize> = vec![0; self.gates.gates.len()];
        let mut qubit_depths: HashMap<Qubit, usize> = HashMap::new();
        for gate in self.gates.gates.iter() {
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
            gate_depths[gate.id() as usize] = max_depth;
        }

        // Validate Routing
        // -------------------------------
        // Validation Closures
        let path_valid_ends =
            |route: &Route, source: &Location, target: &Location| -> Result<(), ValidationError> {
                let start = route
                    .path
                    .first()
                    .ok_or(RoutingError::InvalidTerminalPathLocations(route.gate.id()))?;
                let start_valid = source % self.arch.width == start % self.arch.width;
                let end = route
                    .path
                    .last()
                    .ok_or(RoutingError::InvalidTerminalPathLocations(route.gate.id()))?;
                let end_valid = target - 1 == *end || target + 1 == *end;
                if !start_valid || !end_valid {
                    Err(RoutingError::InvalidTerminalPathLocations(route.gate.id()))?;
                }
                Ok(())
            };
        let valid_route_loc = |route: &Route, loc: &Location| -> Result<(), ValidationError> {
            if *loc >= self.arch.height * self.arch.width
                || used_qubit_locations.contains(loc)
                || magic_locations.contains(loc)
            {
                Err(RoutingError::InvalidPathLocations(route.gate.id()))?;
            }
            Ok(())
        };
        // Validate layers
        // Reuse qubit depths to track layers
        qubit_depths.clear();
        for layer in self.steps.iter() {
            // Validate physical route
            let mut used_locations: HashSet<Location> = HashSet::new();
            layer
                .iter()
                .try_for_each(|r| -> Result<(), ValidationError> {
                    // Make sure path locations are valid
                    r.path
                        .iter()
                        .try_for_each(|loc| -> Result<(), ValidationError> {
                            valid_route_loc(r, loc)?;
                            if used_locations.contains(loc) {
                                Err(RoutingError::OverlappingPathLocations(r.gate.id()))?;
                            }
                            used_locations.insert(*loc);
                            Ok(())
                        })?;
                    let (q0, maybe_q1) = r.gate.get_qubits();
                    // Make sure gate depth is equal to current qubit depth
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
                    // Ensure max depth equals gate circuit depth
                    if max_depth != gate_depths[r.gate.id() as usize] {
                        Err(RoutingError::UnequalGateDepths(r.gate.id()))?;
                    }
                    // Make sure path ends are valid
                    if let Some(q1) = maybe_q1 {
                        let source = self.map.get(&q0).unwrap();
                        let target = self.map.get(&q1).unwrap();
                        path_valid_ends(r, source, target)?;
                    } else {
                        let source = self.map.get(&q0).unwrap();
                        if !self
                            .arch
                            .magic_locations
                            .iter()
                            .any(|target| path_valid_ends(r, source, target).is_ok())
                        {
                            Err(RoutingError::InvalidTerminalPathLocations(r.gate.id()))?;
                        }
                    }
                    Ok(())
                })?;
        }
        // assert!(false);
        Ok(())
    }
}
