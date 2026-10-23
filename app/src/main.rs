//! filming-better: upscale + frame interpolation CLI (TensorRT + NVDEC/NVENC).

mod cli;
mod io;
mod models;
mod native;
mod pipeline;
mod report;

fn main() {
    let code = cli::main_entry();
    std::process::exit(code);
}
