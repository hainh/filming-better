//! `models verify`: sha256 of every installed engine.
use super::store::ModelStore;
use anyhow::{bail, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Streaming sha256 of a file (lowercase hex), calling `on_bytes(n)` after every chunk.
pub fn sha256_file_with(path: &Path, mut on_bytes: impl FnMut(u64)) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        on_bytes(n as u64);
    }
    Ok(hex::encode(h.finalize()))
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    sha256_file_with(path, |_| {})
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyStatus {
    Ok,
    /// Wrong size or wrong hash; the string explains.
    Mismatch(String),
    /// Not installed in any search dir.
    Missing,
}

#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub id: String,
    pub arch: String,
    pub file: String,
    pub path: Option<PathBuf>,
    pub status: VerifyStatus,
}

/// Verify every engine listed in the manifest. Engines present in several search dirs are
/// reported once per copy; absent engines are reported `Missing`.
pub fn verify_store(store: &ModelStore, mut on_bytes: impl FnMut(&str, u64)) -> Vec<VerifyResult> {
    let mut out = Vec::new();
    for m in &store.manifest().models {
        for e in &m.engines {
            let mut found = false;
            for d in store.dirs() {
                let p = d.join(&e.file);
                let Ok(md) = std::fs::metadata(&p) else { continue };
                if !md.is_file() {
                    continue;
                }
                found = true;
                let status = if e.size != 0 && md.len() != e.size {
                    VerifyStatus::Mismatch(format!(
                        "size {} bytes, expected {}",
                        md.len(),
                        e.size
                    ))
                } else {
                    match sha256_file_with(&p, |n| on_bytes(&e.file, n)) {
                        Ok(h) if h.eq_ignore_ascii_case(&e.sha256) => VerifyStatus::Ok,
                        Ok(h) => VerifyStatus::Mismatch(format!(
                            "sha256 {}, expected {}",
                            h, e.sha256
                        )),
                        Err(err) => VerifyStatus::Mismatch(format!("cannot read file: {err}")),
                    }
                };
                out.push(VerifyResult {
                    id: m.id.clone(),
                    arch: e.arch.clone(),
                    file: e.file.clone(),
                    path: Some(p),
                    status,
                });
            }
            if !found {
                out.push(VerifyResult {
                    id: m.id.clone(),
                    arch: e.arch.clone(),
                    file: e.file.clone(),
                    path: None,
                    status: VerifyStatus::Missing,
                });
            }
        }
    }
    out
}

/// Human-readable multi-line text for results (also used by `report`).
pub fn format_verify(results: &[VerifyResult]) -> String {
    let mut s = String::new();
    for r in results {
        let (tag, detail) = match &r.status {
            VerifyStatus::Ok => ("OK      ", r.path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()),
            VerifyStatus::Mismatch(why) => (
                "MISMATCH",
                format!("{} ({})", r.path.as_ref().map(|p| p.display().to_string()).unwrap_or_default(), why),
            ),
            VerifyStatus::Missing => ("MISSING ", "not installed".to_string()),
        };
        s.push_str(&format!("{} {} [{}] {}\n", tag, r.id, r.arch, detail));
    }
    let (ok, bad, missing) = counts(results);
    s.push_str(&format!("{} ok, {} mismatch, {} missing\n", ok, bad, missing));
    s
}

pub fn counts(results: &[VerifyResult]) -> (usize, usize, usize) {
    let ok = results.iter().filter(|r| r.status == VerifyStatus::Ok).count();
    let missing = results.iter().filter(|r| r.status == VerifyStatus::Missing).count();
    let bad = results.len() - ok - missing;
    (ok, bad, missing)
}

/// Print OK/MISMATCH/MISSING for each engine; `Err` if any file is damaged.
pub fn cmd_verify() -> Result<()> {
    let store = ModelStore::open()?;
    verify_and_print(&store)
}

pub fn verify_and_print(store: &ModelStore) -> Result<()> {
    let results = verify_store(store, |_, _| {});
    print!("{}", format_verify(&results));
    let (_, bad, _) = counts(&results);
    if bad > 0 {
        let ids: Vec<&str> = results
            .iter()
            .filter(|r| matches!(r.status, VerifyStatus::Mismatch(_)))
            .map(|r| r.id.as_str())
            .collect();
        bail!(
            "{} engine file(s) failed verification. Fix: run `filming-better models pull {} --force`",
            bad,
            ids.join(" ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::*;
    use std::fs;

    #[test]
    fn sha256_known_vector() {
        let t = TempDir::new("sha");
        fs::write(t.join("a"), b"abc").unwrap();
        assert_eq!(
            sha256_file(&t.join("a")).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut total = 0;
        sha256_file_with(&t.join("a"), |n| total += n).unwrap();
        assert_eq!(total, 3);
        assert!(sha256_file(&t.join("nope")).is_err());
    }

    #[test]
    fn verify_ok_mismatch_missing() {
        let t = TempDir::new("ver");
        let (json, blobs) = sample_manifest("u/");
        fs::create_dir_all(t.join("d")).unwrap();
        // span ampere+: correct; rife: same size, flipped content; sm75: absent
        fs::write(t.join("d").join(&blobs[0].0), &blobs[0].1).unwrap();
        let mut bad = blobs[1].1.clone();
        bad[0] ^= 0xff;
        fs::write(t.join("d").join(&blobs[1].0), &bad).unwrap();
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json).unwrap();
        let r = verify_store(&s, |_, _| {});
        assert_eq!(r.len(), 3);
        let by = |id: &str, arch: &str| r.iter().find(|x| x.id == id && x.arch == arch).unwrap();
        assert_eq!(by("span_x2", "ampere+").status, VerifyStatus::Ok);
        assert_eq!(by("span_x2", "sm75").status, VerifyStatus::Missing);
        assert!(matches!(by("rife_v4", "ampere+").status, VerifyStatus::Mismatch(ref m) if m.contains("sha256")));
        assert_eq!(counts(&r), (1, 1, 1));
        let txt = format_verify(&r);
        assert!(txt.contains("OK       span_x2 [ampere+]"), "{txt}");
        assert!(txt.contains("MISMATCH rife_v4"), "{txt}");
        assert!(txt.contains("MISSING  span_x2 [sm75]"), "{txt}");
        assert!(txt.contains("1 ok, 1 mismatch, 1 missing"), "{txt}");
        let e = verify_and_print(&s).unwrap_err();
        assert!(e.to_string().contains("rife_v4"), "{e}");
    }

    #[test]
    fn verify_wrong_size_and_all_ok() {
        let t = TempDir::new("ver2");
        let (json, blobs) = sample_manifest("u/");
        fs::create_dir_all(t.join("d")).unwrap();
        for (n, b) in &blobs {
            fs::write(t.join("d").join(n), b).unwrap();
        }
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json).unwrap();
        assert!(verify_and_print(&s).is_ok());
        fs::write(t.join("d/span_x2.sm75.engine"), b"x").unwrap();
        let r = verify_store(&s, |_, _| {});
        assert!(r.iter().any(|x| matches!(&x.status, VerifyStatus::Mismatch(m) if m.contains("size"))));
        assert!(verify_and_print(&s).is_err());
    }
}
