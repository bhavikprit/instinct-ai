use reflex_rs::SemanticVectorEncoder;
use std::io::{self, Read};

fn main() {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer).expect("Failed to read stdin");

    let encoder = SemanticVectorEncoder::new();
    let lines: Vec<&str> = buffer.lines().collect();

    print!("[");
    for (i, line) in lines.iter().enumerate() {
        let vec = encoder.encode(line);
        print!("[");
        for (j, val) in vec.iter().enumerate() {
            print!("{:.8}", val);
            if j + 1 < vec.len() {
                print!(",");
            }
        }
        print!("]");
        if i + 1 < lines.len() {
            print!(",");
        }
    }
    println!("]");
}
