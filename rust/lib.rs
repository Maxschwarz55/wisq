use pyo3::prelude::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    time::{Duration, Instant},
};

pub mod mapping;
pub mod parser;
pub mod routing;
pub mod structures;
pub mod validate;

#[pymodule]
fn dascot_rs(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(compile_qasm, _m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(text_signature = "(qasm,path, json_path, arch_layout, init_mapping_temp,
init_routing_temp, term_mapping_temp, term_routing_temp, cooling_mapping_ratem
cooling_routing_rate,/)")]
fn compile_qasm(
    qasm_path: String,
    json_path: String,
    arch_layout: String,
    init_mapping_temp: f64,
    init_routing_temp: f64,
    term_mapping_temp: f64,
    term_routing_temp: f64,
    cooling_mapping_rate: f64,
    cooling_routing_rate: f64,
    timeout_secs: u64,

)-> PyResult<()> {

    let path = qasm_path;
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let circuit = parser::parallel_parse_qasm(reader)?;

    let arch = structures::Architecture::square_sparse(circuit.qubits.len(), true);

    // Create timeout deadline
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);

    let m_and_r = mapping::run_simulated_annealing(
        arch,
        &circuit,
        init_mapping_temp,
        init_routing_temp,
        term_mapping_temp,
        term_routing_temp,
        cooling_mapping_rate,
        cooling_routing_rate,
        deadline,
    );

    let out_path = json_path;
    let out_file = File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);
    let json_str = serde_json::to_string_pretty(&m_and_r)
      .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError,
    _>(format!("JSON error: {}", e)))?;

    writer.write_fmt(format_args!("{}", json_str))?;
    Ok(())

}