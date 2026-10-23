//! Command line interface: argument definitions and dispatch.

pub mod config;
pub mod errors;
pub mod gpu_info;
pub mod preview;
pub mod progress;
pub mod run;

use crate::models;
use clap::{Args, Parser, Subcommand};
use errors::{classify, print_report, CliError};
use std::path::PathBuf;

const ABOUT: &str = "Upscale and increase the frame rate of videos on NVIDIA GPUs (TensorRT + NVDEC/NVENC)";

const MAIN_HELP: &str = "\
Examples:
  filming-better run --input in.mp4 --output out.mp4 --upscale span_x2 --interp 2 --codec hevc --cq 20
  filming-better run --input \"D:\\clips\\*.mp4\" --output-dir D:\\out --preset balanced
  filming-better preview --input in.mp4 --at 00:01:23 --upscale span_x2 --out preview.png
  filming-better gpu-info
  filming-better models list

Exit codes: 0 ok, 1 error, 2 bad usage, 3 no NVIDIA GPU, 4 driver too old, 5 out of VRAM,
            6 unsupported codec/format, 7 model/engine missing, 8 some batch files failed,
            130 cancelled (Ctrl+C).";

#[derive(Parser, Debug)]
#[command(name = "filming-better", version, about = ABOUT, after_help = MAIN_HELP, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Upscale / interpolate one or more videos
    Run(run::RunArgs),
    /// Render a before/after PNG of a single frame
    Preview(preview::PreviewArgs),
    /// Show GPU, driver, runtimes and which engines match this machine
    #[command(name = "gpu-info")]
    GpuInfo(GpuInfoArgs),
    /// Manage model engines
    Models {
        #[command(subcommand)]
        action: ModelsCmd,
    },
    /// Bundle logs and system information into a zip for bug reports
    Report(ReportArgs),
}

#[derive(Args, Debug)]
#[command(after_help = "Example:\n  filming-better gpu-info")]
pub struct GpuInfoArgs {
    /// CUDA device index
    #[arg(long, default_value_t = 0)]
    pub device: i32,
}

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// List models and whether they are installed
    List,
    /// Download engines for this GPU
    #[command(after_help = "Examples:\n  filming-better models pull span_x2\n  filming-better models pull rife_v4 --force")]
    Pull {
        /// Model ids (see `models list`); none = everything for this GPU
        ids: Vec<String>,
        /// Download again even if already installed
        #[arg(long)]
        force: bool,
        #[arg(long, default_value_t = 0)]
        device: i32,
    },
    /// Check engine files against the manifest checksums
    Verify,
}

#[derive(Args, Debug)]
#[command(after_help = "Example:\n  filming-better report --out report.zip --log-file run.log")]
pub struct ReportArgs {
    /// Zip file to write
    #[arg(long, default_value = "filming-better-report.zip")]
    pub out: PathBuf,
    /// Include this log file (from `run --log-file`)
    #[arg(long)]
    pub log_file: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
}

/// Create FFmpeg's CUDA device first (see `CudaDevice::shared`); errors surface later in `native::init`.
pub fn prepare_device(device: i32) {
    let _ = crate::io::CudaDevice::shared(device);
}

/// Build the `models::GpuCaps` for a device.
pub fn gpu_caps(device: i32) -> anyhow::Result<models::GpuCaps> {
    prepare_device(device);
    crate::native::init(device)?;
    let g = crate::native::gpu_info(device)?;
    Ok(models::GpuCaps {
        name: g.name,
        cc_major: g.cc_major,
        cc_minor: g.cc_minor,
        driver_version: g.driver_version,
        total_mem: g.total_mem,
    })
}

fn dispatch(cli: Cli) -> anyhow::Result<i32> {
    match cli.command {
        Command::Run(a) => run::cmd_run(a),
        Command::Preview(a) => preview::cmd_preview(a).map(|_| 0),
        Command::GpuInfo(a) => gpu_info::cmd_gpu_info(a.device).map(|_| 0),
        Command::Models { action } => {
            match action {
                ModelsCmd::List => {
                    // listing works without a GPU; just no "matching" column then
                    let gpu = gpu_caps(0).ok();
                    models::cmd_list(gpu.as_ref())?;
                }
                ModelsCmd::Pull { ids, force, device } => {
                    let gpu = gpu_caps(device)?;
                    models::cmd_pull(&ids, &gpu, force)?;
                }
                ModelsCmd::Verify => models::cmd_verify()?,
            }
            Ok(0)
        }
        Command::Report(a) => {
            let info = gpu_info::system_info(a.device);
            crate::report::create_report(&a.out, a.log_file.as_deref(), &info)?;
            println!("wrote {}", a.out.display());
            Ok(0)
        }
    }
}

/// Parse arguments, run, return the process exit code.
pub fn main_entry() -> i32 {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            let code = if e.use_stderr() { errors::EXIT_USAGE } else { 0 };
            let _ = e.print();
            return code;
        }
    };
    crate::io::set_log_level(16);
    match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            let r = classify(&e);
            if r.code != errors::EXIT_CANCELLED {
                print_report(&r);
            }
            r.code
        }
    }
}

/// Convenience for usage errors.
pub fn usage<T>(msg: impl Into<String>) -> anyhow::Result<T> {
    Err(CliError::Usage(msg.into()).into())
}
