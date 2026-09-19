use reflex_rs::CompiledInstinct;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: compiled_parity <model_path> [query1] [query2] ...");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let model = match CompiledInstinct::from_file(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    let queries = if args.len() > 2 {
        args[2..].to_vec()
    } else {
        Vec::new()
    };

    print!("[");
    for (i, q) in queries.iter().enumerate() {
        let res = model.predict(q);
        print!("{{\"query\":");
        print!("\"{}\",", q.replace('\\', "\\\\").replace('"', "\\\""));
        print!("\"selected\":\"{}\",", res.selected);
        print!("\"distribution\":{{");
        for (j, (opt, prob)) in res.distribution.iter().enumerate() {
            print!("\"{}\":{:.6}", opt, prob);
            if j + 1 < res.distribution.len() {
                print!(",");
            }
        }
        print!("}}}}");
        if i + 1 < queries.len() {
            print!(",");
        }
    }
    println!("]");
}
