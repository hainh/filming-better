//! `models pull`: download the engines for this GPU, verify sha256, install atomically.
use super::manifest::EngineEntry;
use super::select::{arch_candidates, best_engine};
use super::store::ModelStore;
use super::verify::sha256_file;
use super::{GpuCaps, ModelError};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Where `pull` installs: `$FB_MODELS_DIR` if set, else `%LOCALAPPDATA%\filming-better\models`.
pub fn pull_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("FB_MODELS_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("filming-better")
        .join("models")
}

#[derive(Debug, Clone)]
pub struct PullOptions {
    /// Destination directory (engines + manifest.json).
    pub dest: PathBuf,
    /// Overrides the manifest's `base_url` (mirrors, tests). Env `FB_MODELS_URL` in `cmd_pull`.
    pub base_url: Option<String>,
    pub force: bool,
    /// Show progress bars.
    pub progress: bool,
}

/// Download engines for `gpu`. Empty `ids` = every model that has an engine for this GPU.
/// Returns the installed file paths (including ones that were already up to date).
pub fn pull_with(
    store: &ModelStore,
    ids: &[String],
    gpu: &GpuCaps,
    opts: &PullOptions,
) -> Result<Vec<PathBuf>> {
    let arch = arch_candidates(gpu)?.join("/");
    let mut wanted: Vec<(&super::manifest::ModelEntry, &EngineEntry)> = Vec::new();
    if ids.is_empty() {
        for m in &store.manifest().models {
            match best_engine(m, gpu) {
                Ok(e) => wanted.push((m, e)),
                Err(_) => eprintln!("note: model '{}' has no {} engine; skipped", m.id, arch),
            }
        }
    } else {
        for id in ids {
            let (m, e) = store.select_engine(id, gpu)?;
            if !wanted.iter().any(|(w, _)| w.id == m.id) {
                wanted.push((m, e));
            }
        }
    }
    if wanted.is_empty() {
        anyhow::bail!("nothing to download for this GPU ({arch})");
    }

    std::fs::create_dir_all(&opts.dest)
        .with_context(|| format!("cannot create models directory {}", opts.dest.display()))?;
    write_atomic(&opts.dest.join("manifest.json"), store.manifest_raw().as_bytes())
        .context("cannot write manifest.json")?;

    let mut client: Option<reqwest::blocking::Client> = None;
    let mut out = Vec::new();
    for (m, e) in wanted {
        let target = opts.dest.join(&e.file);
        if !opts.force && is_valid_install(&target, e) {
            println!("{} [{}]: already installed and verified", m.id, e.arch);
            out.push(target);
            continue;
        }
        let url = store.manifest().engine_url(opts.base_url.as_deref(), e);
        if client.is_none() {
            client = Some(build_client(&url)?);
        }
        let label = format!("{} [{}]", m.id, e.arch);
        let part = opts.dest.join(format!("{}.part", e.file));
        let res = download_to(client.as_ref().unwrap(), &url, &part, e, &label, opts.progress);
        if let Err(err) = res {
            let _ = std::fs::remove_file(&part);
            return Err(err);
        }
        std::fs::rename(&part, &target).with_context(|| {
            format!(
                "cannot move {} to {} (is the engine in use by another filming-better process?)",
                part.display(),
                target.display()
            )
        })?;
        println!("{}: installed {}", label, target.display());
        out.push(target);
    }
    Ok(out)
}

/// Download engines for this GPU into [`pull_dir`].
pub fn cmd_pull(ids: &[String], gpu: &GpuCaps, force: bool) -> Result<()> {
    let store = ModelStore::open()?;
    let opts = PullOptions {
        dest: pull_dir(),
        base_url: std::env::var("FB_MODELS_URL").ok().filter(|s| !s.is_empty()),
        force,
        progress: true,
    };
    let files = pull_with(&store, ids, gpu, &opts)?;
    println!("{} engine(s) ready in {}", files.len(), opts.dest.display());
    Ok(())
}

fn is_valid_install(path: &Path, e: &EngineEntry) -> bool {
    match std::fs::metadata(path) {
        Ok(md) if md.is_file() && (e.size == 0 || md.len() == e.size) => sha256_file(path)
            .map(|h| h.eq_ignore_ascii_case(&e.sha256))
            .unwrap_or(false),
        _ => false,
    }
}

fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("json.part");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

fn build_client(first_url: &str) -> Result<reqwest::blocking::Client> {
    let mut b = reqwest::blocking::Client::builder()
        .user_agent(concat!("filming-better/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .timeout(None::<Duration>);
    let local = first_url.starts_with("http://127.0.0.1")
        || first_url.starts_with("http://localhost")
        || first_url.starts_with("http://[::1]");
    if local {
        b = b.no_proxy();
    }
    b.build().context("cannot initialize HTTP client")
}

fn download_to(
    client: &reqwest::blocking::Client,
    url: &str,
    part: &Path,
    e: &EngineEntry,
    label: &str,
    progress: bool,
) -> Result<()> {
    let mut resp = client.get(url).send().map_err(|err| {
        ModelError::Download(format!(
            "cannot reach {url}: {err}. Check your internet connection or proxy settings"
        ))
    })?;
    let status = resp.status();
    if !status.is_success() {
        let hint = match status.as_u16() {
            404 => " (this engine is not published at that address yet; check `base_url` in the manifest or set FB_MODELS_URL)",
            401 | 403 => " (access denied)",
            429 => " (rate limited, try again later)",
            500..=599 => " (server error, try again later)",
            _ => "",
        };
        return Err(ModelError::Download(format!("HTTP {} for {}{}", status.as_u16(), url, hint)).into());
    }
    let total = resp.content_length();
    if let (Some(t), true) = (total, e.size != 0) {
        if t != e.size {
            return Err(ModelError::Download(format!(
                "server reports {t} bytes for {url}, manifest expects {}",
                e.size
            ))
            .into());
        }
    }

    let bar = if progress {
        let pb = indicatif::ProgressBar::new(total.unwrap_or(e.size));
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "{msg} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
            )
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
            .progress_chars("=> "),
        );
        pb.set_message(label.to_string());
        pb
    } else {
        indicatif::ProgressBar::hidden()
    };

    let file = std::fs::File::create(part)
        .with_context(|| format!("cannot create {}", part.display()))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    loop {
        let n = resp
            .read(&mut buf)
            .map_err(|err| ModelError::Download(format!("connection lost while downloading {url}: {err}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        w.write_all(&buf[..n])
            .with_context(|| format!("cannot write {} (disk full?)", part.display()))?;
        done += n as u64;
        bar.set_position(done);
    }
    w.flush()?;
    w.into_inner().map_err(|e| e.into_error())?.sync_all().ok();
    bar.finish_and_clear();

    if e.size != 0 && done != e.size {
        return Err(ModelError::Download(format!(
            "incomplete download of {url}: got {done} bytes, expected {}",
            e.size
        ))
        .into());
    }
    let got = hex::encode(hasher.finalize());
    if !got.eq_ignore_ascii_case(&e.sha256) {
        return Err(ModelError::Download(format!(
            "sha256 mismatch for {}: got {}, expected {}. The file was discarded; try again or report this",
            e.file, got, e.sha256
        ))
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::*;
    use std::collections::HashMap;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Tiny HTTP server: path -> (status, body). Unknown paths give 404.
    struct Server {
        url: String,
        hits: Arc<AtomicUsize>,
    }

    fn serve(routes: HashMap<String, (u16, Vec<u8>)>) -> Server {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", l.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let h2 = hits.clone();
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let routes = routes.clone();
                let hits = h2.clone();
                std::thread::spawn(move || {
                    let mut req = Vec::new();
                    let mut b = [0u8; 1024];
                    while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                        match s.read(&mut b) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => req.extend_from_slice(&b[..n]),
                        }
                    }
                    let text = String::from_utf8_lossy(&req).to_string();
                    let path = text.split_whitespace().nth(1).unwrap_or("/").to_string();
                    hits.fetch_add(1, Ordering::SeqCst);
                    let (code, body) = routes
                        .get(&path)
                        .cloned()
                        .unwrap_or((404, b"not found".to_vec()));
                    let head = format!(
                        "HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        code,
                        body.len()
                    );
                    let _ = s.write_all(head.as_bytes());
                    let _ = s.write_all(&body);
                });
            }
        });
        Server { url, hits }
    }

    fn routes_for(blobs: &[(String, Vec<u8>)]) -> HashMap<String, (u16, Vec<u8>)> {
        blobs
            .iter()
            .map(|(n, b)| (format!("/{}", n), (200u16, b.clone())))
            .collect()
    }

    fn opts(dest: PathBuf, url: &str, force: bool) -> PullOptions {
        PullOptions { dest, base_url: Some(url.to_string()), force, progress: false }
    }

    #[test]
    fn pull_installs_verifies_and_skips() {
        let t = TempDir::new("pull");
        let (json, blobs) = sample_manifest("http://unused.invalid/");
        let srv = serve(routes_for(&blobs));
        let store = ModelStore::with_manifest_text(vec![t.join("m")], &json).unwrap();
        let dest = t.join("m");

        let files = pull_with(&store, &["span".to_string()], &gpu(8, 6), &opts(dest.clone(), &srv.url, false)).unwrap();
        assert_eq!(files, vec![dest.join("span_x2.ampere+.engine")]);
        assert_eq!(std::fs::read(&files[0]).unwrap(), blobs[0].1);
        assert!(dest.join("manifest.json").is_file());
        assert!(!dest.join("span_x2.ampere+.engine.part").exists());
        assert!(std::fs::read_dir(&dest).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".part")));
        assert_eq!(srv.hits.load(Ordering::SeqCst), 1);

        // second time: already installed, no request
        pull_with(&store, &["span_x2".to_string()], &gpu(8, 6), &opts(dest.clone(), &srv.url, false)).unwrap();
        assert_eq!(srv.hits.load(Ordering::SeqCst), 1);
        // force downloads again
        pull_with(&store, &["span_x2".to_string()], &gpu(8, 6), &opts(dest.clone(), &srv.url, true)).unwrap();
        assert_eq!(srv.hits.load(Ordering::SeqCst), 2);
        // damaged install is repaired without force
        std::fs::write(dest.join("span_x2.ampere+.engine"), vec![0u8; 1000]).unwrap();
        pull_with(&store, &["span_x2".to_string()], &gpu(8, 6), &opts(dest.clone(), &srv.url, false)).unwrap();
        assert_eq!(std::fs::read(dest.join("span_x2.ampere+.engine")).unwrap(), blobs[0].1);
        assert_eq!(srv.hits.load(Ordering::SeqCst), 3);

        // the installed copy now resolves via the store
        assert!(store.resolve("span_x2", &gpu(8, 6)).is_ok());
    }

    #[test]
    fn pull_all_respects_arch() {
        let t = TempDir::new("pullall");
        let (json, blobs) = sample_manifest("x/");
        let srv = serve(routes_for(&blobs));
        let store = ModelStore::with_manifest_text(vec![], &json).unwrap();
        let f = pull_with(&store, &[], &gpu(8, 9), &opts(t.join("a"), &srv.url, false)).unwrap();
        assert_eq!(f.len(), 2);
        let f = pull_with(&store, &[], &gpu(7, 5), &opts(t.join("b"), &srv.url, false)).unwrap();
        assert_eq!(f, vec![t.join("b/span_x2.sm75.engine")]);
        assert!(pull_with(&store, &[], &gpu(6, 1), &opts(t.join("c"), &srv.url, false)).is_err());
        // explicit model without engine for this arch is an error
        let e = pull_with(&store, &["rife_v4".to_string()], &gpu(7, 5), &opts(t.join("d"), &srv.url, false)).unwrap_err();
        assert!(e.to_string().contains("no engine for sm75"), "{e}");
    }

    #[test]
    fn pull_downloads_best_engine_for_gpu() {
        let t = TempDir::new("pullsm");
        let (json, mut blobs) = sample_manifest("x/");
        let (json, extra) = add_sm86_engine(&json);
        blobs.push(extra);
        let srv = serve(routes_for(&blobs));
        let store = ModelStore::with_manifest_text(vec![], &json).unwrap();
        // cc 8.6: sm86 for span_x2, ampere+ for rife_v4 (no sm86 build)
        let f = pull_with(&store, &[], &gpu(8, 6), &opts(t.join("a"), &srv.url, false)).unwrap();
        assert_eq!(f, vec![t.join("a/span_x2.sm86.engine"), t.join("a/rife_v4.ampere+.engine")]);
        // explicit id picks the same engine
        let f = pull_with(&store, &["span".to_string()], &gpu(8, 6), &opts(t.join("b"), &srv.url, false)).unwrap();
        assert_eq!(f, vec![t.join("b/span_x2.sm86.engine")]);
        // cc 8.9: foreign sm86 is not downloaded, ampere+ is
        let f = pull_with(&store, &["span".to_string()], &gpu(8, 9), &opts(t.join("c"), &srv.url, false)).unwrap();
        assert_eq!(f, vec![t.join("c/span_x2.ampere+.engine")]);
        // cc 7.5 unchanged
        let f = pull_with(&store, &[], &gpu(7, 5), &opts(t.join("d"), &srv.url, false)).unwrap();
        assert_eq!(f, vec![t.join("d/span_x2.sm75.engine")]);
    }

    #[test]
    fn pull_rejects_corrupt_download() {
        let t = TempDir::new("pullbad");
        let (json, blobs) = sample_manifest("x/");
        let mut bad = blobs[0].1.clone();
        bad[10] ^= 1; // same size, wrong content
        let mut r = routes_for(&blobs);
        r.insert("/span_x2.ampere+.engine".into(), (200, bad));
        let srv = serve(r);
        let store = ModelStore::with_manifest_text(vec![], &json).unwrap();
        let e = pull_with(&store, &["span_x2".into()], &gpu(8, 6), &opts(t.join("m"), &srv.url, false)).unwrap_err();
        assert!(e.to_string().contains("sha256 mismatch"), "{e}");
        assert!(!t.join("m/span_x2.ampere+.engine").exists());
        assert!(!t.join("m/span_x2.ampere+.engine.part").exists());
    }

    #[test]
    fn pull_rejects_wrong_length() {
        let t = TempDir::new("pulllen");
        let (json, blobs) = sample_manifest("x/");
        let mut r = routes_for(&blobs);
        r.insert("/span_x2.ampere+.engine".into(), (200, vec![1u8; 10]));
        let srv = serve(r);
        let store = ModelStore::with_manifest_text(vec![], &json).unwrap();
        let e = pull_with(&store, &["span_x2".into()], &gpu(8, 6), &opts(t.join("m"), &srv.url, false)).unwrap_err();
        assert!(e.to_string().contains("manifest expects 1000"), "{e}");
        assert!(!t.join("m/span_x2.ampere+.engine.part").exists());
    }

    #[test]
    fn pull_http_errors_are_clear() {
        let t = TempDir::new("pullerr");
        let (json, blobs) = sample_manifest("x/");
        let mut r = routes_for(&blobs);
        r.insert("/span_x2.ampere+.engine".into(), (503, b"busy".to_vec()));
        let srv = serve(r);
        let store = ModelStore::with_manifest_text(vec![], &json).unwrap();
        let e = pull_with(&store, &["span_x2".into()], &gpu(8, 6), &opts(t.join("m"), &srv.url, false)).unwrap_err();
        assert!(e.to_string().contains("HTTP 503"), "{e}");
        // 404 for a file the server does not have
        let srv2 = serve(HashMap::new());
        let e = pull_with(&store, &["span_x2".into()], &gpu(8, 6), &opts(t.join("m"), &srv2.url, false)).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("HTTP 404") && s.contains("not published"), "{s}");
        // connection refused
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let dead = format!("http://{}/", l.local_addr().unwrap());
        drop(l);
        let e = pull_with(&store, &["span_x2".into()], &gpu(8, 6), &opts(t.join("m"), &dead, false)).unwrap_err();
        assert!(e.to_string().contains("cannot reach"), "{e}");
        assert!(matches!(e.downcast_ref::<ModelError>(), Some(ModelError::Download(_))));
    }

    #[test]
    fn pull_dir_honors_env() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let old = std::env::var_os("FB_MODELS_DIR");
        std::env::set_var("FB_MODELS_DIR", "Z:/custom");
        assert_eq!(pull_dir(), PathBuf::from("Z:/custom"));
        std::env::remove_var("FB_MODELS_DIR");
        assert!(pull_dir().ends_with("filming-better/models") || pull_dir().ends_with("filming-better\\models"));
        if let Some(o) = old {
            std::env::set_var("FB_MODELS_DIR", o);
        }
    }
}
