//! `filming-better report`: bundle diagnostics into a zip for bug reports.
//!
//! Contents: system info, OS version, FB_* environment, the log file (tail), every manifest
//! found, installed engines with sizes, `models verify` output and the config file.
//! The user's home directory is replaced by `%USERPROFILE%` everywhere.

use crate::models::store::ModelStore;
use crate::models::verify::{format_verify, verify_store};
use anyhow::{Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Max bytes of the log file included (the end of the file is kept).
pub const MAX_LOG_BYTES: u64 = 5 * 1024 * 1024;

/// Everything `create_report_with` needs besides the arguments of `create_report`
/// (separated so tests can run without touching the real machine state).
pub struct ReportInputs<'a> {
    pub store: Result<&'a ModelStore, String>,
    pub env: Vec<(String, String)>,
    pub home: Option<PathBuf>,
    pub os_version: String,
    pub config_file: Option<PathBuf>,
    pub run_verify: bool,
}

/// Collect a diagnostic zip at `out_zip`.
pub fn create_report(out_zip: &Path, log_file: Option<&Path>, system_info: &str) -> Result<()> {
    let store = ModelStore::open();
    let inputs = ReportInputs {
        store: store.as_ref().map_err(|e| format!("{:#}", e)),
        env: std::env::vars().collect(),
        home: dirs::home_dir(),
        os_version: os_version(),
        config_file: crate::cli::config::default_config_path().filter(|p| p.is_file()),
        run_verify: true,
    };
    create_report_with(out_zip, log_file, system_info, &inputs)
}

pub fn os_version() -> String {
    let mut s = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
    if cfg!(windows) {
        if let Ok(o) = std::process::Command::new("cmd").args(["/C", "ver"]).output() {
            let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !v.is_empty() {
                s = format!("{}\n{}", v, s);
            }
        }
    }
    s
}

pub fn create_report_with(
    out_zip: &Path,
    log_file: Option<&Path>,
    system_info: &str,
    inputs: &ReportInputs,
) -> Result<()> {
    use zip::write::SimpleFileOptions;

    if let Some(parent) = out_zip.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let file = std::fs::File::create(out_zip)
        .with_context(|| format!("cannot create report file {}", out_zip.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let home = inputs.home.as_deref();
    let red = |s: &str| redact(s, home);

    let add = |zip: &mut zip::ZipWriter<std::fs::File>, name: &str, content: &str| -> Result<()> {
        zip.start_file(name, opts)?;
        zip.write_all(content.as_bytes())?;
        Ok(())
    };

    add(
        &mut zip,
        "README.txt",
        &format!(
            "filming-better diagnostic report (version {})\n\
             Generated: unix time {}\n\n\
             system_info.txt   GPU/driver/system information\n\
             os.txt            operating system version\n\
             env.txt           FB_* environment variables\n\
             log/              log file (last {} MiB)\n\
             manifests/        model manifests found on this machine\n\
             engines.txt       installed engines and sizes\n\
             models_verify.txt `models verify` output (sha256 of every engine)\n\
             config.toml       your config file (if any)\n\n\
             Your home directory is replaced by %USERPROFILE%.\n",
            env!("CARGO_PKG_VERSION"),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            MAX_LOG_BYTES / (1024 * 1024)
        ),
    )?;
    add(&mut zip, "system_info.txt", &red(system_info))?;
    add(&mut zip, "os.txt", &red(&inputs.os_version))?;

    // Environment: FB_* only; anything that looks like a secret is masked.
    let mut env_txt = String::new();
    let mut env: Vec<&(String, String)> = inputs
        .env
        .iter()
        .filter(|(k, _)| k.to_ascii_uppercase().starts_with("FB_"))
        .collect();
    env.sort();
    for (k, v) in env {
        let ku = k.to_ascii_uppercase();
        let secret = ["TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD"].iter().any(|w| ku.contains(w));
        env_txt.push_str(&format!("{}={}\n", k, if secret { "<redacted>".to_string() } else { red(v) }));
    }
    if env_txt.is_empty() {
        env_txt.push_str("(no FB_* variables set)\n");
    }
    add(&mut zip, "env.txt", &env_txt)?;

    // Log file (tail).
    if let Some(lf) = log_file {
        match read_tail(lf, MAX_LOG_BYTES) {
            Ok((bytes, truncated)) => {
                let mut text = String::new();
                if truncated {
                    text.push_str(&format!("[... log truncated, showing the last {} bytes ...]\n", MAX_LOG_BYTES));
                }
                text.push_str(&String::from_utf8_lossy(&bytes));
                let name = lf
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "log.txt".into());
                add(&mut zip, &format!("log/{}", sanitize(&name)), &red(&text))?;
            }
            Err(e) => add(
                &mut zip,
                "log/NOT_INCLUDED.txt",
                &red(&format!("could not read log file {}: {}\n", lf.display(), e)),
            )?,
        }
    }

    // Models.
    match &inputs.store {
        Ok(store) => {
            let mut index = String::new();
            index.push_str("embedded: manifests/embedded.json (built into the executable)\n");
            add(&mut zip, "manifests/embedded.json", crate::models::store::EMBEDDED_MANIFEST)?;
            for (i, p) in store.manifest_files().iter().enumerate() {
                let name = format!("manifests/found_{}.json", i + 1);
                index.push_str(&format!("{}: {}\n", name, p.display()));
                match std::fs::read_to_string(p) {
                    Ok(t) => add(&mut zip, &name, &red(&t))?,
                    Err(e) => add(&mut zip, &name, &format!("cannot read: {}\n", e))?,
                }
            }
            index.push_str(&format!("in use: {:?}\n", store.manifest_source()));
            add(&mut zip, "manifests/index.txt", &red(&index))?;

            let mut eng = String::new();
            eng.push_str("Search directories (in order):\n");
            for d in store.dirs() {
                eng.push_str(&format!("  {} {}\n", d.display(), if d.is_dir() { "" } else { "(does not exist)" }));
            }
            eng.push_str("\nInstalled engines:\n");
            let inst = store.installed_engines();
            if inst.is_empty() {
                eng.push_str("  (none)\n");
            }
            for e in inst {
                eng.push_str(&format!(
                    "  {} [{}] {} bytes{} - {}\n",
                    e.id,
                    e.arch,
                    e.size,
                    if e.size_ok { "" } else { " (SIZE MISMATCH)" },
                    e.path.display()
                ));
            }
            add(&mut zip, "engines.txt", &red(&eng))?;

            let v = if inputs.run_verify {
                format_verify(&verify_store(store, |_, _| {}))
            } else {
                "(skipped)\n".to_string()
            };
            add(&mut zip, "models_verify.txt", &red(&v))?;
        }
        Err(e) => {
            add(&mut zip, "engines.txt", &red(&format!("model store unavailable: {}\n", e)))?;
            add(&mut zip, "manifests/embedded.json", crate::models::store::EMBEDDED_MANIFEST)?;
        }
    }

    if let Some(cfg) = &inputs.config_file {
        if let Ok(t) = std::fs::read_to_string(cfg) {
            add(&mut zip, "config.toml", &red(&t))?;
        }
    }

    zip.finish().context("cannot finish zip")?;
    Ok(())
}

/// Replace the home directory with `%USERPROFILE%` (case-insensitive; `\`, `/` and JSON-escaped
/// `\\` spellings).
pub fn redact(text: &str, home: Option<&Path>) -> String {
    let Some(home) = home else { return text.to_string() };
    let h = home.to_string_lossy().trim_end_matches(['\\', '/']).to_string();
    if h.len() < 4 {
        return text.to_string(); // "C:" or "/" would wreck the text
    }
    let forms = [h.replace('\\', "\\\\"), h.clone(), h.replace('\\', "/")];
    let mut out = text.to_string();
    for f in forms {
        out = replace_ci(&out, &f, "%USERPROFILE%");
    }
    out
}

fn replace_ci(text: &str, needle: &str, with: &str) -> String {
    if needle.is_empty() {
        return text.to_string();
    }
    // ASCII lowercasing preserves byte offsets, so indexes map back onto `text`.
    let lt = text.to_ascii_lowercase();
    let ln = needle.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(pos) = lt[i..].find(&ln) {
        out.push_str(&text[i..i + pos]);
        out.push_str(with);
        i += pos + ln.len();
    }
    out.push_str(&text[i..]);
    out
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect()
}

/// Last `max` bytes of a file; second value = whether it was truncated.
fn read_tail(path: &Path, max: u64) -> std::io::Result<(Vec<u8>, bool)> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let truncated = len > max;
    if truncated {
        f.seek(SeekFrom::Start(len - max))?;
    }
    let mut buf = Vec::new();
    f.take(max).read_to_end(&mut buf)?;
    Ok((buf, truncated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::*;
    use std::collections::HashMap;
    use std::fs;

    fn read_zip(p: &Path) -> HashMap<String, String> {
        let mut z = zip::ZipArchive::new(fs::File::open(p).unwrap()).unwrap();
        let mut m = HashMap::new();
        for i in 0..z.len() {
            let mut f = z.by_index(i).unwrap();
            let mut s = String::new();
            f.read_to_string(&mut s).unwrap();
            m.insert(f.name().to_string(), s);
        }
        m
    }

    #[test]
    fn redaction_forms() {
        let home = Path::new("C:\\Users\\Alice");
        let t = "a C:\\Users\\Alice\\x b c:/users/alice/y c C:\\\\Users\\\\Alice\\\\z d C:\\Users\\Bob";
        let r = redact(t, Some(home));
        assert_eq!(
            r,
            "a %USERPROFILE%\\x b %USERPROFILE%/y c %USERPROFILE%\\\\z d C:\\Users\\Bob"
        );
        assert_eq!(redact("nothing", Some(home)), "nothing");
        assert_eq!(redact("x", None), "x");
        assert_eq!(redact("C:\\a", Some(Path::new("C:\\"))), "C:\\a");
        // trailing separator on home
        assert_eq!(redact("C:\\Users\\Alice\\q", Some(Path::new("C:\\Users\\Alice\\"))), "%USERPROFILE%\\q");
        // non-ascii text survives
        assert_eq!(redact("Tiếng Việt C:\\Users\\Alice", Some(home)), "Tiếng Việt %USERPROFILE%");
    }

    #[test]
    fn report_contents() {
        let t = TempDir::new("report");
        let home = t.join("Users/Alice");
        let models = home.join("models");
        fs::create_dir_all(&models).unwrap();
        let (json, blobs) = sample_manifest("u/");
        fs::write(models.join("manifest.json"), &json).unwrap();
        fs::write(models.join(&blobs[0].0), &blobs[0].1).unwrap();
        let mut bad = blobs[1].1.clone();
        bad[0] ^= 1;
        fs::write(models.join(&blobs[1].0), &bad).unwrap();
        let store = ModelStore::from_dirs(vec![models.clone()]).unwrap();

        let log = home.join("fb.log");
        fs::write(&log, format!("started in {}\nline2\n", home.display())).unwrap();
        let cfg = home.join("config.toml");
        fs::write(&cfg, "cq = 20\n").unwrap();

        let inputs = ReportInputs {
            store: Ok(&store),
            env: vec![
                ("FB_MODELS_DIR".into(), models.display().to_string()),
                ("FB_API_TOKEN".into(), "s3cret".into()),
                ("PATH".into(), "C:\\secret\\path".into()),
                ("USERNAME".into(), "Alice".into()),
            ],
            home: Some(home.clone()),
            os_version: "Windows test 10.0".into(),
            config_file: Some(cfg),
            run_verify: true,
        };
        let out = t.join("out/report.zip");
        create_report_with(&out, Some(&log), &format!("GPU: RTX test\nhome={}\n", home.display()), &inputs).unwrap();
        let z = read_zip(&out);

        for k in [
            "README.txt", "system_info.txt", "os.txt", "env.txt", "log/fb.log", "manifests/embedded.json",
            "manifests/found_1.json", "manifests/index.txt", "engines.txt", "models_verify.txt", "config.toml",
        ] {
            assert!(z.contains_key(k), "missing {k}: {:?}", z.keys());
        }
        assert!(z["system_info.txt"].contains("GPU: RTX test"));
        assert!(z["system_info.txt"].contains("%USERPROFILE%"));
        assert!(!z["system_info.txt"].contains("Alice"));
        assert!(z["os.txt"].contains("Windows test"));
        assert!(z["env.txt"].contains("FB_MODELS_DIR=%USERPROFILE%"));
        assert!(z["env.txt"].contains("FB_API_TOKEN=<redacted>"));
        assert!(!z["env.txt"].contains("PATH") && !z["env.txt"].contains("USERNAME"));
        assert!(!z["env.txt"].contains("s3cret"));
        assert!(z["log/fb.log"].contains("started in %USERPROFILE%"));
        assert!(!z["log/fb.log"].contains("Alice"));
        assert!(z["manifests/found_1.json"].contains("\"span_x2\""));
        assert!(z["engines.txt"].contains("span_x2 [ampere+] 1000 bytes"));
        assert!(z["engines.txt"].contains("rife_v4 [ampere+] 2500 bytes"));
        assert!(!z["engines.txt"].contains("Alice"));
        assert!(z["models_verify.txt"].contains("OK       span_x2 [ampere+]"));
        assert!(z["models_verify.txt"].contains("MISMATCH rife_v4"));
        assert!(z["models_verify.txt"].contains("MISSING  span_x2 [sm75]"));
        assert_eq!(z["config.toml"], "cq = 20\n");
        assert!(!z.values().any(|v| v.contains("Alice")), "home leaked");
    }

    #[test]
    fn report_without_log_store_or_with_big_log() {
        let t = TempDir::new("report2");
        let inputs = ReportInputs {
            store: Err("boom".into()),
            env: vec![],
            home: None,
            os_version: "x".into(),
            config_file: None,
            run_verify: false,
        };
        let out = t.join("r.zip");
        create_report_with(&out, Some(&t.join("missing.log")), "info", &inputs).unwrap();
        let z = read_zip(&out);
        assert!(z["log/NOT_INCLUDED.txt"].contains("could not read"));
        assert!(z["engines.txt"].contains("model store unavailable: boom"));
        assert!(z["env.txt"].contains("no FB_*"));
        assert!(!z.contains_key("config.toml"));

        // big log: only the tail is kept
        let big = t.join("big.log");
        let mut data = vec![b'a'; (MAX_LOG_BYTES as usize) + 1000];
        let n = data.len();
        data[n - 4..].copy_from_slice(b"END\n");
        fs::write(&big, &data).unwrap();
        let store = ModelStore::with_embedded(vec![t.join("none")]).unwrap();
        let inputs2 = ReportInputs { store: Ok(&store), run_verify: true, ..inputs };
        create_report_with(&out, Some(&big), "info", &inputs2).unwrap();
        let z = read_zip(&out);
        let l = &z["log/big.log"];
        assert!(l.contains("log truncated"));
        assert!(l.ends_with("END\n"));
        assert!(l.len() < data.len());
        assert!(z["models_verify.txt"].contains("MISSING"));
    }
}
