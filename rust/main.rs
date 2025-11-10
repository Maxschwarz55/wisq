use dascot_rs::{
    mapping::run_simulated_annealing, parser::parallel_parse_qasm, structures::Architecture
};
use std::{
    env,
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Write},
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("Usage: cargo run <path.qasm> <path.json>");
        return Ok(());
    }

    let path = &args[1];
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    println!("Parsing");
    let mut circuit: dascot_rs::structures::Circuit = parallel_parse_qasm(reader)?;
    circuit.sort_gates();

    println!("Layout");
    let arch = Architecture::square_sparse(circuit.qubits.len(), true);


    println!("Mapping and Routing");
    //Set time out to 1000. This can be changed
    let deadline = Instant::now() + Duration::from_secs(1000);
    //Simulated annealing parameters hardcoded. This can be changed
    let m_and_r = run_simulated_annealing(arch, &circuit, 100.0, 50.0, 1.0, 1.0, 0.5, 0.5, deadline);
    //let map = m_and_r.map;
    
    let out_path = &args[2];
    let out_file = File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);
    writer.write_fmt(format_args!("{}", serde_json::to_string_pretty(&m_and_r)?))?;

    Ok(())
}
