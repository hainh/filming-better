//! `filming-better run`: argument resolution, input expansion, batch loop, summary.

use super::config::{self, RunDefaults};
use super::errors::*;
use super::progress::{fmt_duration, ProgressMode, Reporter};
use super::usage;
use crate::io::encode::VideoCodec;
use crate::models::{ModelStore, ResolvedEngine};
use crate::native;
use crate::pipeline::{self, Ctl, JobSpec, JobStats, Order};
use anyhow::{Context, Result};
use clap::Args;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub const DEFAULT_SCENE_THRESHOLD: f32 = 0.10;
pub const DEFAULT_NAME_TEMPLATE: &str = "{stem}_fb.{ext}";
pub const DEFAULT_INTERP_MODEL: &str = "rife_v4";
pub const VIDEO_EXTS: [&str; 13] =
    ["mp4", "mkv", "mov", "m4v", "avi", "webm", "ts", "m2ts", "mts", "wmv", "flv", "mpg", "mpeg"];

const RUN_HELP: &str = "\
Examples:
  Upscale 2x and double the frame rate, HEVC at CQ 20:
    filming-better run --input in.mp4 --output out.mp4 --upscale span_x2 --interp 2 --codec hevc --cq 20

  Whole folder with a preset, results next to a chosen directory:
    filming-better run --input D:\\clips --output-dir D:\\out --preset balanced --skip-existing

  Wildcards (quote them so the shell does not expand them), JSON progress for scripts:
    filming-better run --input \"D:\\clips\\*.mp4\" --output-dir D:\\out --progress json

  Frame interpolation only, 3x:
    filming-better run --input in.mp4 --output out.mp4 --upscale none --interp 3

Precedence of options: explicit flags > --preset > config file > built-in defaults.
Config file: --config <file> or %APPDATA%\\filming-better\\config.toml (TOML keys: preset, upscale,
interp, codec, cq, order, scene_threshold, output_dir, name_template, tile, log_file).
Presets: fast = span_x2 + interp 2 + hevc cq 24, balanced = span_x2 + interp 2 + hevc cq 20,
quality = animevideov3_x4 + interp 2 + hevc cq 18.";

#[derive(Args, Debug, Clone, Default)]
#[command(after_help = RUN_HELP)]
pub struct RunArgs {
    /// Input file, directory or glob pattern (repeatable)
    #[arg(short, long, required = true, num_args = 1.., value_name = "PATH")]
    pub input: Vec<String>,
    /// Output file (single input only)
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,
    /// Output directory for batch processing
    #[arg(long, value_name = "DIR")]
    pub output_dir: Option<PathBuf>,
    /// Output file name pattern: {stem}, {ext}, {name}
    #[arg(long, value_name = "TEMPLATE")]
    pub name_template: Option<String>,
    /// Upscale model id (see `models list`) or `none`
    #[arg(long, value_name = "MODEL|none")]
    pub upscale: Option<String>,
    /// Frame interpolation factor: 1 = off, 2, 3 or 4 = output frames per input frame
    #[arg(long, value_name = "N")]
    pub interp: Option<u32>,
    /// Interpolation model id
    #[arg(long, value_name = "MODEL", default_value = DEFAULT_INTERP_MODEL)]
    pub interp_model: String,
    /// Output codec: h264, hevc or av1 (av1 needs an RTX 40-series GPU or newer)
    #[arg(long, value_name = "CODEC")]
    pub codec: Option<String>,
    /// Constant quality for NVENC, 0-51 (lower = better, bigger file)
    #[arg(long, value_name = "N")]
    pub cq: Option<u32>,
    /// Preset: fast, balanced or quality
    #[arg(long, value_name = "NAME")]
    pub preset: Option<String>,
    /// Stage order: interp-first (faster, default) or upscale-first
    #[arg(long, value_name = "ORDER")]
    pub order: Option<String>,
    /// Scene change threshold 0..1 (frames after a cut are repeated, not interpolated)
    #[arg(long, value_name = "F")]
    pub scene_threshold: Option<f32>,
    /// Maximum tile edge in pixels for upscaling large frames
    #[arg(long, value_name = "PX")]
    pub tile: Option<u32>,
    /// Skip inputs whose output file already exists
    #[arg(long)]
    pub skip_existing: bool,
    /// Progress output: bar (default), json (JSON lines on stdout) or none
    #[arg(long, value_name = "MODE", default_value = "bar")]
    pub progress: String,
    /// Write a detailed log to this file
    #[arg(long, value_name = "FILE")]
    pub log_file: Option<PathBuf>,
    /// Config file (TOML)
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// CUDA device index
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    /// NVENC preset p1 (fastest) .. p7 (best)
    #[arg(long, value_name = "P", default_value = "p5")]
    pub nvenc_preset: String,
    /// Decode on the CPU instead of NVDEC
    #[arg(long)]
    pub cpu_decode: bool,
}

impl RunArgs {
    /// The options the user typed (None = not given), in the shape the config module merges.
    pub fn explicit(&self) -> RunDefaults {
        RunDefaults {
            preset: self.preset.clone(),
            upscale: self.upscale.clone(),
            interp: self.interp,
            codec: self.codec.clone(),
            cq: self.cq,
            order: self.order.clone(),
            scene_threshold: self.scene_threshold,
            output_dir: self.output_dir.clone(),
            name_template: self.name_template.clone(),
            tile: self.tile,
            log_file: self.log_file.clone(),
        }
    }
}

/// Fully resolved options.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub upscale: Option<String>,
    pub interp: u32,
    pub interp_model: String,
    pub codec: VideoCodec,
    pub cq: u32,
    pub order: Order,
    pub scene_threshold: f32,
    pub tile: Option<u32>,
    pub nvenc_preset: String,
    pub output_dir: Option<PathBuf>,
    pub name_template: String,
    pub log_file: Option<PathBuf>,
}

/// Merge explicit flags, preset, config file and built-in defaults, then validate.
pub fn resolve_settings(args: &RunArgs, cfg: &RunDefaults) -> Result<Settings> {
    let merged = config::resolve_run(&args.explicit(), cfg)?;
    let upscale = merged.upscale.filter(|u| {
        let l = u.trim().to_ascii_lowercase();
        !(l.is_empty() || l == "none" || l == "off" || l == "0")
    });
    let interp = merged.interp.unwrap_or(config::DEFAULT_INTERP);
    if !(1..=4).contains(&interp) {
        return usage(format!("--interp must be 1, 2, 3 or 4 (got {interp})"));
    }
    let codec_s = merged.codec.unwrap_or_else(|| config::DEFAULT_CODEC.to_string());
    let Some(codec) = VideoCodec::parse(&codec_s) else {
        return usage(format!("unknown codec '{codec_s}' (use h264, hevc or av1)"));
    };
    let cq = merged.cq.unwrap_or(config::DEFAULT_CQ);
    if cq > 51 {
        return usage(format!("--cq must be between 0 and 51 (got {cq})"));
    }
    let order_s = merged.order.unwrap_or_else(|| "interp-first".to_string());
    let Some(order) = Order::parse(&order_s) else {
        return usage(format!("unknown order '{order_s}' (use interp-first or upscale-first)"));
    };
    let scene_threshold = merged.scene_threshold.unwrap_or(DEFAULT_SCENE_THRESHOLD);
    if !(0.0..=1.0).contains(&scene_threshold) {
        return usage(format!("--scene-threshold must be between 0 and 1 (got {scene_threshold})"));
    }
    let nv = args.nvenc_preset.to_ascii_lowercase();
    if !(nv.len() == 2 && nv.starts_with('p') && matches!(nv.as_bytes()[1], b'1'..=b'7')) {
        return usage(format!("--nvenc-preset must be p1..p7 (got {})", args.nvenc_preset));
    }
    if let Some(t) = merged.tile {
        if t < 64 {
            return usage(format!("--tile must be at least 64 (got {t})"));
        }
    }
    if ProgressMode::parse(&args.progress).is_none() {
        return usage(format!("--progress must be bar, json or none (got {})", args.progress));
    }
    Ok(Settings {
        upscale,
        interp,
        interp_model: args.interp_model.clone(),
        codec,
        cq,
        order,
        scene_threshold,
        tile: merged.tile,
        nvenc_preset: nv,
        output_dir: merged.output_dir,
        name_template: merged.name_template.unwrap_or_else(|| DEFAULT_NAME_TEMPLATE.to_string()),
        log_file: merged.log_file,
    })
}

// ---------------------------------------------------------------------------------------
// input / output path handling
// ---------------------------------------------------------------------------------------

fn is_video_file(p: &Path) -> bool {
    p.extension()
        .map(|e| VIDEO_EXTS.contains(&e.to_string_lossy().to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn has_glob_chars(s: &str) -> bool {
    s.contains(['*', '?', '['])
}

/// Expand `--input` values (file, directory or glob) into a sorted, de-duplicated file list.
pub fn expand_inputs(inputs: &[String]) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for raw in inputs {
        let p = PathBuf::from(raw);
        if p.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(&p)
                .with_context(|| format!("cannot read directory '{raw}'"))?
                .flatten()
                .map(|e| e.path())
                .filter(|f| f.is_file() && is_video_file(f))
                .collect();
            found.sort();
            if found.is_empty() {
                return usage(format!("no video files found in directory '{raw}'"));
            }
            out.extend(found);
        } else if p.is_file() {
            out.push(p);
        } else if has_glob_chars(raw) {
            let mut found: Vec<PathBuf> = glob::glob(raw)
                .map_err(|e| anyhow::Error::new(CliError::Usage(format!("bad pattern '{raw}': {e}"))))?
                .flatten()
                .filter(|f| f.is_file())
                .collect();
            found.sort();
            if found.is_empty() {
                return usage(format!("no files match '{raw}'"));
            }
            out.extend(found);
        } else {
            return usage(format!("input '{raw}' does not exist"));
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| seen.insert(p.clone()));
    Ok(out)
}

/// Apply `{stem}`, `{ext}`, `{name}` to a template.
pub fn apply_template(template: &str, input: &Path) -> Result<String> {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into());
    let ext = input.extension().map(|s| s.to_string_lossy().to_ascii_lowercase()).unwrap_or_else(|| "mp4".into());
    let name = input.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let s = template.replace("{stem}", &stem).replace("{ext}", &ext).replace("{name}", &name);
    if let (Some(a), Some(b)) = (s.find('{'), s.find('}')) {
        if a < b {
            return usage(format!("unknown placeholder {} in --name-template (use {{stem}}, {{ext}}, {{name}})", &s[a..=b]));
        }
    }
    // a template without an extension keeps the input's container
    if Path::new(&s).extension().is_none() {
        return Ok(format!("{s}.{ext}"));
    }
    Ok(s)
}

/// Output path for one input.
pub fn output_for(input: &Path, single_output: Option<&Path>, out_dir: Option<&Path>, template: &str) -> Result<PathBuf> {
    if let Some(o) = single_output {
        return Ok(o.to_path_buf());
    }
    let name = apply_template(template, input)?;
    let dir = match out_dir {
        Some(d) => d.to_path_buf(),
        None => input.parent().map(|p| p.to_path_buf()).unwrap_or_default(),
    };
    Ok(dir.join(name))
}

// ---------------------------------------------------------------------------------------
// cancellation
// ---------------------------------------------------------------------------------------

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static CURRENT_CTL: Mutex<Option<Arc<Ctl>>> = Mutex::new(None);

fn install_ctrlc() {
    let _ = ctrlc::set_handler(|| {
        if CANCEL_REQUESTED.swap(true, Ordering::SeqCst) {
            std::process::exit(EXIT_CANCELLED);
        }
        eprintln!("\ncancelling: finishing the output file (press Ctrl+C again to quit immediately)");
        if let Some(c) = CURRENT_CTL.lock().unwrap().as_ref() {
            c.cancel();
        }
    });
}

pub fn cancel_requested() -> bool {
    CANCEL_REQUESTED.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------------------

pub fn init_logging(log_file: Option<&Path>) -> Result<()> {
    use tracing_subscriber::EnvFilter;
    match log_file {
        Some(p) => {
            if let Some(d) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(d).ok();
            }
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .with_context(|| format!("cannot open log file '{}'", p.display()))?;
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(Mutex::new(f))
                .try_init();
        }
        None => {
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
            let _ = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------------------

enum Status {
    Done(JobStats),
    Skipped,
    Failed(Report),
    Cancelled(JobStats),
}

struct Row {
    input: PathBuf,
    output: PathBuf,
    status: Status,
}

/// Resolve both engines for this GPU.
pub fn resolve_engines(settings: &Settings, gpu: &crate::models::GpuCaps) -> Result<(Option<ResolvedEngine>, Option<ResolvedEngine>)> {
    let need = settings.upscale.is_some() || settings.interp > 1;
    if !need {
        return Ok((None, None));
    }
    let store = ModelStore::open()?;
    let up = match &settings.upscale {
        Some(id) => Some(store.resolve(id, gpu)?),
        None => None,
    };
    let ip = if settings.interp > 1 { Some(store.resolve(&settings.interp_model, gpu)?) } else { None };
    if let Some(e) = &up {
        if e.kind != crate::models::ModelKind::Upscale {
            return usage(format!("'{}' is not an upscale model", e.id));
        }
    }
    if let Some(e) = &ip {
        if e.kind != crate::models::ModelKind::Interp {
            return usage(format!("'{}' is not an interpolation model", e.id));
        }
    }
    Ok((up, ip))
}

pub fn cmd_run(args: RunArgs) -> Result<i32> {
    let cfg = config::load_config(args.config.as_deref())?;
    let settings = resolve_settings(&args, &cfg)?;
    init_logging(settings.log_file.as_deref())?;
    let mode = ProgressMode::parse(&args.progress).unwrap_or(ProgressMode::Bar);

    let inputs = expand_inputs(&args.input)?;
    if args.output.is_some() && inputs.len() > 1 {
        return usage("--output can only be used with a single input; use --output-dir for several files");
    }
    // output paths + sanity checks
    let mut jobs: Vec<(PathBuf, PathBuf)> = Vec::new();
    for inp in &inputs {
        let out = output_for(inp, args.output.as_deref(), settings.output_dir.as_deref(), &settings.name_template)?;
        if same_file(inp, &out) {
            return usage(format!("output '{}' would overwrite the input", out.display()));
        }
        jobs.push((inp.clone(), out));
    }
    {
        let mut seen = std::collections::HashSet::new();
        for (_, o) in &jobs {
            if !seen.insert(o.clone()) {
                return usage(format!("two inputs would write the same output file '{}'", o.display()));
            }
        }
    }
    if settings.upscale.is_none() && settings.interp == 1 {
        tracing::warn!("neither upscaling nor interpolation selected: the video is only re-encoded");
    }

    // GPU + engines (fail early, before touching any file)
    super::prepare_device(args.device);
    let drv = native::init(args.device)?;
    tracing::info!("CUDA driver {drv}");
    let gpu = super::gpu_caps(args.device)?;
    let (up_engine, ip_engine) = resolve_engines(&settings, &gpu)?;
    if settings.codec == VideoCodec::Av1 && !crate::io::encode::nvenc_supports(VideoCodec::Av1, gpu.cc_major, gpu.cc_minor) {
        return Err(crate::io::IoError::Unsupported(format!(
            "{} (compute capability {}.{}) has no AV1 hardware encoder; AV1 needs an RTX 40-series or newer. Use --codec hevc or --codec h264",
            gpu.name, gpu.cc_major, gpu.cc_minor
        ))
        .into());
    }

    install_ctrlc();
    let total = jobs.len();
    let mut rows: Vec<Row> = Vec::new();
    for (idx, (inp, out)) in jobs.into_iter().enumerate() {
        if cancel_requested() {
            break;
        }
        if args.skip_existing && out.exists() {
            if mode != ProgressMode::Json {
                eprintln!("[{}/{}] skipping {} (output exists)", idx + 1, total, inp.display());
            }
            rows.push(Row { input: inp, output: out, status: Status::Skipped });
            continue;
        }
        if mode != ProgressMode::Json {
            eprintln!("[{}/{}] {} -> {}", idx + 1, total, inp.display(), out.display());
        }
        let spec = JobSpec {
            input: inp.clone(),
            output: out.clone(),
            upscale: up_engine.clone(),
            interp_engine: ip_engine.clone(),
            interp_factor: settings.interp,
            order: settings.order,
            codec: settings.codec,
            cq: settings.cq,
            nvenc_preset: settings.nvenc_preset.clone(),
            scene_threshold: settings.scene_threshold,
            tile: settings.tile,
            device: args.device,
            cpu_decode: args.cpu_decode,
        };
        let ctl = Ctl::new(args.device);
        *CURRENT_CTL.lock().unwrap() = Some(ctl.clone());
        if cancel_requested() {
            ctl.cancel();
        }
        let label = format!("{:>3}/{:<3}", idx + 1, total);
        let rep = Reporter::start(ctl.clone(), mode, label);
        let res = pipeline::run_job(&spec, &ctl);
        rep.finish();
        *CURRENT_CTL.lock().unwrap() = None;
        let status = match res {
            Ok(st) if st.cancelled => Status::Cancelled(st),
            Ok(st) => {
                if mode != ProgressMode::Json {
                    eprintln!(
                        "  done: {}x{} @ {:.2} fps, {} frames in {} ({:.1} fps)",
                        st.out_w,
                        st.out_h,
                        st.out_fps,
                        st.frames_out,
                        fmt_duration(st.elapsed.as_secs_f64()),
                        st.fps()
                    );
                    for s in &st.skipped_streams {
                        eprintln!("  warning: {s}");
                    }
                }
                Status::Done(st)
            }
            Err(e) => {
                let r = classify(&e);
                if mode == ProgressMode::Json {
                    eprintln!("error: {}", r.message);
                } else {
                    print_report(&r);
                }
                tracing::debug!("{} failed: {:#}", inp.display(), e);
                Status::Failed(r)
            }
        };
        rows.push(Row { input: inp, output: out, status });
    }

    let cancelled = cancel_requested();
    let failed: Vec<&Row> = rows.iter().filter(|r| matches!(r.status, Status::Failed(_))).collect();
    if total > 1 || cancelled {
        print_summary(&rows, total, mode);
    }
    if cancelled {
        return Ok(EXIT_CANCELLED);
    }
    if failed.is_empty() {
        return Ok(EXIT_OK);
    }
    if total == 1 {
        if let Status::Failed(r) = &failed[0].status {
            return Ok(r.code);
        }
    }
    Ok(EXIT_BATCH_FAILED)
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

fn print_summary(rows: &[Row], total: usize, mode: ProgressMode) {
    let mut s = String::new();
    s.push_str(&format!("\n{:<40} {:<10} {:>8} {:>9} {:>8}\n", "file", "status", "frames", "time", "fps"));
    s.push_str(&format!("{}\n", "-".repeat(79)));
    let (mut ok, mut fail, mut skip) = (0, 0, 0);
    for r in rows {
        let name = r.input.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let name = if name.chars().count() > 39 { format!("{}~", name.chars().take(38).collect::<String>()) } else { name };
        let (status, frames, time, fps) = match &r.status {
            Status::Done(st) => {
                ok += 1;
                ("ok", st.frames_out.to_string(), fmt_duration(st.elapsed.as_secs_f64()), format!("{:.1}", st.fps()))
            }
            Status::Cancelled(st) => ("cancelled", st.frames_out.to_string(), fmt_duration(st.elapsed.as_secs_f64()), format!("{:.1}", st.fps())),
            Status::Skipped => {
                skip += 1;
                ("skipped", "-".into(), "-".into(), "-".into())
            }
            Status::Failed(_) => {
                fail += 1;
                ("FAILED", "-".into(), "-".into(), "-".into())
            }
        };
        s.push_str(&format!("{name:<40} {status:<10} {frames:>8} {time:>9} {fps:>8}\n"));
        if let Status::Failed(rep) = &r.status {
            s.push_str(&format!("    {}\n", rep.message.lines().next().unwrap_or("")));
        }
        let _ = &r.output;
    }
    s.push_str(&format!("{ok} ok, {fail} failed, {skip} skipped, {} not started (of {total})\n", total - rows.len()));
    if mode == ProgressMode::Json {
        eprint!("{s}");
    } else {
        print!("{s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    fn parse(args: &[&str]) -> RunArgs {
        let mut v = vec!["filming-better", "run"];
        v.extend_from_slice(args);
        match Cli::try_parse_from(v).unwrap().command {
            Command::Run(a) => a,
            _ => panic!("not run"),
        }
    }

    #[test]
    fn parses_full_command_line() {
        let a = parse(&[
            "--input", "in.mp4", "--output", "out.mkv", "--upscale", "span_x2", "--interp", "2", "--codec", "hevc",
            "--cq", "20",
        ]);
        assert_eq!(a.input, vec!["in.mp4"]);
        assert_eq!(a.output.as_deref(), Some(Path::new("out.mkv")));
        assert_eq!(a.interp, Some(2));
        assert_eq!(a.cq, Some(20));
        assert_eq!(a.progress, "bar");
    }

    #[test]
    fn repeatable_inputs() {
        let a = parse(&["--input", "a.mp4", "--input", "b.mp4", "-i", "c.mp4"]);
        assert_eq!(a.input.len(), 3);
    }

    #[test]
    fn missing_input_is_usage_error() {
        let e = Cli::try_parse_from(["filming-better", "run", "--output", "x.mp4"]).unwrap_err();
        assert!(e.use_stderr());
    }

    #[test]
    fn explicit_beats_preset_beats_config() {
        let cfg = RunDefaults { cq: Some(30), codec: Some("h264".into()), interp: Some(4), ..Default::default() };
        // preset fast = span_x2/2/hevc/24; explicit --cq wins over the preset, preset beats config
        let a = parse(&["--input", "x.mp4", "--preset", "fast", "--cq", "18"]);
        let s = resolve_settings(&a, &cfg).unwrap();
        assert_eq!(s.cq, 18);
        assert_eq!(s.codec, VideoCodec::Hevc);
        assert_eq!(s.interp, 2);
        assert_eq!(s.upscale.as_deref(), Some("span_x2"));
        // no preset: config beats built-in
        let a = parse(&["--input", "x.mp4"]);
        let s = resolve_settings(&a, &cfg).unwrap();
        assert_eq!((s.cq, s.interp, s.codec), (30, 4, VideoCodec::H264));
        assert_eq!(s.upscale, None);
        // built-in defaults
        let s = resolve_settings(&a, &RunDefaults::default()).unwrap();
        assert_eq!((s.cq, s.interp, s.codec, s.order), (20, 2, VideoCodec::Hevc, Order::InterpFirst));
        assert_eq!(s.scene_threshold, DEFAULT_SCENE_THRESHOLD);
        assert_eq!(s.name_template, DEFAULT_NAME_TEMPLATE);
    }

    #[test]
    fn upscale_none_overrides_preset() {
        let a = parse(&["--input", "x.mp4", "--preset", "balanced", "--upscale", "none"]);
        let s = resolve_settings(&a, &RunDefaults::default()).unwrap();
        assert_eq!(s.upscale, None);
        assert_eq!(s.interp, 2);
    }

    #[test]
    fn validation_errors_are_usage() {
        for bad in [
            vec!["--input", "x.mp4", "--interp", "5"],
            vec!["--input", "x.mp4", "--interp", "0"],
            vec!["--input", "x.mp4", "--codec", "vp9"],
            vec!["--input", "x.mp4", "--cq", "99"],
            vec!["--input", "x.mp4", "--order", "sideways"],
            vec!["--input", "x.mp4", "--scene-threshold", "2"],
            vec!["--input", "x.mp4", "--preset", "ultra"],
            vec!["--input", "x.mp4", "--progress", "xml"],
            vec!["--input", "x.mp4", "--tile", "8"],
        ] {
            let a = parse(&bad);
            let e = resolve_settings(&a, &RunDefaults::default()).unwrap_err();
            let r = classify(&e);
            // unknown preset is reported by the config module as a plain error
            assert!(r.code == EXIT_USAGE || r.code == EXIT_GENERIC, "{bad:?} -> {r:?}");
            assert!(!r.message.is_empty());
        }
    }

    #[test]
    fn name_templates() {
        let p = Path::new("D:/clips/My Clip.MP4");
        assert_eq!(apply_template("{stem}_fb.{ext}", p).unwrap(), "My Clip_fb.mp4");
        assert_eq!(apply_template("{name}.out.mkv", p).unwrap(), "My Clip.MP4.out.mkv");
        assert_eq!(apply_template("{stem}_up", p).unwrap(), "My Clip_up.mp4");
        assert!(apply_template("{stem}_{bogus}.mp4", p).is_err());
    }

    #[test]
    fn output_paths() {
        let inp = Path::new("D:/clips/a.mp4");
        let o = output_for(inp, None, Some(Path::new("D:/out")), "{stem}_fb.{ext}").unwrap();
        assert_eq!(o, Path::new("D:/out").join("a_fb.mp4"));
        let o = output_for(inp, Some(Path::new("x.mkv")), Some(Path::new("D:/out")), "{stem}").unwrap();
        assert_eq!(o, Path::new("x.mkv"));
        let o = output_for(inp, None, None, "{stem}_fb.{ext}").unwrap();
        assert_eq!(o, Path::new("D:/clips").join("a_fb.mp4"));
    }

    #[test]
    fn expands_directory_and_glob() {
        let dir = std::env::temp_dir().join(format!("fb_expand_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["b.mp4", "a.mkv", "notes.txt", "c.MOV"] {
            std::fs::write(dir.join(n), b"x").unwrap();
        }
        let v = expand_inputs(&[dir.to_string_lossy().into_owned()]).unwrap();
        let names: Vec<String> = v.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["a.mkv", "b.mp4", "c.MOV"]);
        let pat = format!("{}/*.mp4", dir.to_string_lossy().replace('\\', "/"));
        let v = expand_inputs(&[pat]).unwrap();
        assert_eq!(v.len(), 1);
        assert!(expand_inputs(&[format!("{}/*.avi", dir.to_string_lossy().replace('\\', "/"))]).is_err());
        assert!(expand_inputs(&["definitely_missing_file.mp4".to_string()]).is_err());
        // duplicates collapse
        let f = dir.join("b.mp4").to_string_lossy().into_owned();
        assert_eq!(expand_inputs(&[f.clone(), f]).unwrap().len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
