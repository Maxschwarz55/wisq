use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;
use std::{
    collections::HashMap,
    io::{self, BufRead},
};

use crate::structures::{Circuit, Gate, Qubit};

pub fn parallel_parse_qasm<R: BufRead>(reader: R) -> io::Result<Circuit> {
    let mut offset: u32 = 0;
    let mut register_groups: HashMap<String, u32> = HashMap::new();

    // For capturing the qubit register names
    let qreg_re = Regex::new(r"^(qreg)\s+([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\];$").unwrap();
    let mut gate_lines: Vec<String> = Vec::new();
    for line in reader.lines().flatten() {
        if let Some(caps) = qreg_re.captures(&line) {
            let name = caps[2].to_string();
            let size: u32 = caps[3].parse().unwrap();
            register_groups.insert(name, offset);
            offset += size;
        } else {
            gate_lines.push(line);
        }
    }

    let keys = register_groups
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join("|");
    // Regex for capturing cx gates
    let cx_re = Regex::new(&format!(
        r"^(cx)\s+({})\[(\d+)\],\s*({})\[(\d+)\];$",
        keys, keys
    ))
    .unwrap();
    // Regex for capturing t or tdg gates
    let t_re = Regex::new(&format!(r"^(t|tdg)\s+({})\[(\d+)\];$", keys)).unwrap();
    let gates: Vec<Gate> = gate_lines
        .par_iter()
        .filter_map(|line| {
            if let Some(caps) = cx_re.captures(line) {
                let q0 = Qubit::new(caps[3].parse::<u32>().unwrap() + register_groups[&caps[2]]);
                let q1 = Qubit::new(caps[5].parse::<u32>().unwrap() + register_groups[&caps[4]]);

                Some(Gate::cx(0, q0, q1))
            } else if let Some(caps) = t_re.captures(line) {
                let q0 = caps[3].parse::<u32>().unwrap() + register_groups[&caps[2]];
                match caps.get(1).unwrap().as_str() {
                    "t" => Some(Gate::t(0, q0)),
                    "tdg" => Some(Gate::tdg(0, q0)),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect();
    let mut circ = Circuit::new();
    gates.into_iter().enumerate().for_each(|(i, mut g)| {
        g.set_id(i as u32);
        circ.add_gate(g)
    });
    Ok(circ)
}
