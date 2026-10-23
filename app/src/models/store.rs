//! Local model store: search dirs, manifest loading, engine resolution.
use super::manifest::{EngineEntry, Manifest, ModelEntry};
use super::select::{best_engine, match_model_id, ranked_engines};
use super::{GpuCaps, ModelError, ResolvedEngine};
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Manifest compiled into the binary (repo `models/manifest.json`).
pub const EMBEDDED_MANIFEST: &str = include_str!("../../../models/manifest.json");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestSource {
    Embedded,
    File(PathBuf),
}

/// Result of looking for an engine file in the search dirs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Located {
    Found(PathBuf),
    /// A file with the right name exists but its size differs from the manifest.
    BadSize { path: PathBuf, actual: u64 },
    Missing,
}

/// An engine file present on disk (extra; used by report/list).
#[derive(Debug, Clone)]
pub struct InstalledEngine {
    pub id: String,
    pub arch: String,
    pub file: String,
    pub path: PathBuf,
    pub size: u64,
    pub size_ok: bool,
}

#[derive(Debug, Clone)]
pub struct ModelStore {
    dirs: Vec<PathBuf>,
    manifest: Manifest,
    manifest_raw: String,
    source: ManifestSource,
}

/// Walk up from `start` looking for `<ancestor>/models/engines` (dev checkout layout).
pub fn find_dev_models_dir(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .map(|a| a.join("models").join("engines"))
        .find(|p| p.is_dir())
}

/// Build the search list: `$FB_MODELS_DIR`, `<exe_dir>/models`, `%LOCALAPPDATA%/filming-better/models`,
/// and the dev `<repo>/models/engines`. Duplicates removed, order kept.
pub fn search_dirs_from(
    env_dir: Option<PathBuf>,
    exe_dir: Option<&Path>,
    local_data: Option<PathBuf>,
) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Some(d) = env_dir.filter(|d| !d.as_os_str().is_empty()) {
        v.push(d);
    }
    if let Some(e) = exe_dir {
        v.push(e.join("models"));
    }
    if let Some(l) = local_data {
        v.push(l.join("filming-better").join("models"));
    }
    if let Some(e) = exe_dir {
        if let Some(d) = find_dev_models_dir(e) {
            v.push(d);
        }
    }
    let mut out: Vec<PathBuf> = Vec::new();
    for d in v {
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

/// Search dirs for the running process.
pub fn default_search_dirs() -> Vec<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));
    search_dirs_from(
        std::env::var_os("FB_MODELS_DIR").map(PathBuf::from),
        exe_dir.as_deref(),
        dirs::data_local_dir(),
    )
}

/// Places where a `manifest.json` may live for a given search dir. The dev layout
/// keeps engines in `models/engines/` and the manifest one level up.
pub fn manifest_candidates(dir: &Path) -> Vec<PathBuf> {
    let mut v = vec![dir.join("manifest.json")];
    if dir.file_name().map(|n| n == "engines").unwrap_or(false) {
        if let Some(p) = dir.parent() {
            v.push(p.join("manifest.json"));
        }
    }
    v
}

impl ModelStore {
    /// Search order: `$FB_MODELS_DIR`, `<exe_dir>/models`, `%LOCALAPPDATA%/filming-better/models`,
    /// dev `<repo>/models/engines`. Manifest comes from the first dir that has a valid
    /// `manifest.json`, else the embedded copy.
    pub fn open() -> Result<ModelStore> {
        Self::from_dirs(default_search_dirs())
    }

    /// Same as [`open`](Self::open) with explicit dirs (used by tests and tools).
    pub fn from_dirs(dirs: Vec<PathBuf>) -> Result<ModelStore> {
        for d in &dirs {
            for cand in manifest_candidates(d) {
                if !cand.is_file() {
                    continue;
                }
                match std::fs::read_to_string(&cand)
                    .map_err(anyhow::Error::from)
                    .and_then(|t| Manifest::parse(&t).map(|m| (t, m)))
                {
                    Ok((raw, manifest)) => {
                        return Ok(ModelStore {
                            dirs,
                            manifest,
                            manifest_raw: raw,
                            source: ManifestSource::File(cand),
                        })
                    }
                    Err(e) => tracing::warn!("ignoring manifest {}: {:#}", cand.display(), e),
                }
            }
        }
        Self::with_embedded(dirs)
    }

    /// Store that always uses the embedded manifest.
    pub fn with_embedded(dirs: Vec<PathBuf>) -> Result<ModelStore> {
        let manifest = Manifest::parse(EMBEDDED_MANIFEST)?;
        Ok(ModelStore {
            dirs,
            manifest,
            manifest_raw: EMBEDDED_MANIFEST.to_string(),
            source: ManifestSource::Embedded,
        })
    }

    /// Store with an explicit manifest text (tests, tools).
    pub fn with_manifest_text(dirs: Vec<PathBuf>, raw: &str) -> Result<ModelStore> {
        Ok(ModelStore {
            dirs,
            manifest: Manifest::parse(raw)?,
            manifest_raw: raw.to_string(),
            source: ManifestSource::Embedded,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    /// Original manifest text (what `pull` writes next to the engines).
    pub fn manifest_raw(&self) -> &str {
        &self.manifest_raw
    }
    pub fn manifest_source(&self) -> &ManifestSource {
        &self.source
    }
    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// Resolve a user-supplied id (exact, alias, unique prefix) to a manifest model.
    pub fn find_model(&self, id: &str) -> Result<&ModelEntry, ModelError> {
        let ids: Vec<&str> = self.manifest.models.iter().map(|m| m.id.as_str()).collect();
        let real = match_model_id(id, &ids)?;
        Ok(self.manifest.model(real).expect("id came from manifest"))
    }

    /// Model + best engine entry for this GPU according to the manifest alone (no filesystem
    /// access): exact `sm<cc>` engine, else `ampere+` (cc >= 8.0), else `sm75` (cc 7.5).
    /// This is what `pull` downloads.
    pub fn select_engine(
        &self,
        model_id: &str,
        gpu: &GpuCaps,
    ) -> Result<(&ModelEntry, &EngineEntry), ModelError> {
        let m = self.find_model(model_id)?;
        let e = best_engine(m, gpu)?;
        Ok((m, e))
    }

    /// Same preference order as [`select_engine`](Self::select_engine), but an engine that is
    /// installed beats a better-ranked one that is not (so a portable bundle that only ships
    /// `ampere+` keeps working with a manifest that also lists `sm86`). If nothing usable is
    /// installed the best-ranked engine is returned as `Missing` (the `pull` hint downloads it).
    pub fn pick_engine<'a>(
        &self,
        m: &'a ModelEntry,
        gpu: &GpuCaps,
    ) -> Result<(&'a EngineEntry, Located), ModelError> {
        let best = best_engine(m, gpu)?;
        let mut bad: Option<(&EngineEntry, Located)> = None;
        for e in ranked_engines(m, gpu)? {
            match self.locate(e) {
                l @ Located::Found(_) => return Ok((e, l)),
                l @ Located::BadSize { .. } => {
                    if bad.is_none() {
                        bad = Some((e, l));
                    }
                }
                Located::Missing => {}
            }
        }
        Ok(bad.unwrap_or((best, Located::Missing)))
    }

    /// Driver must support at least the CUDA major the engines were built with.
    pub fn check_driver(&self, gpu: &GpuCaps) -> Result<(), ModelError> {
        if gpu.driver_version > 0 && (gpu.driver_version / 1000) < self.manifest.cuda_major as i32 {
            return Err(ModelError::DriverTooOld {
                have: format!("{}.{}", gpu.driver_version / 1000, (gpu.driver_version % 1000) / 10),
                need: self.manifest.cuda_major,
            });
        }
        Ok(())
    }

    /// Check the TensorRT lean runtime version (e.g. "11.4.0.5") against the version the
    /// engines were built with: major versions must match.
    pub fn check_trt_runtime(&self, runtime_version: &str) -> Result<(), ModelError> {
        let rt_major: Option<u32> = runtime_version.trim().split('.').next().and_then(|s| s.parse().ok());
        match (self.manifest.trt_major(), rt_major) {
            (Some(b), Some(r)) if b != r => Err(ModelError::TrtMismatch {
                built: self.manifest.trt_version.clone(),
                runtime: runtime_version.to_string(),
            }),
            _ => Ok(()),
        }
    }

    /// Fast check (existence + size). Manifest `size == 0` means "size unknown" and is not checked.
    pub fn locate(&self, e: &EngineEntry) -> Located {
        let mut bad: Option<Located> = None;
        for d in &self.dirs {
            let p = d.join(&e.file);
            if let Ok(md) = std::fs::metadata(&p) {
                if !md.is_file() {
                    continue;
                }
                if e.size == 0 || md.len() == e.size {
                    return Located::Found(p);
                }
                if bad.is_none() {
                    bad = Some(Located::BadSize { path: p, actual: md.len() });
                }
            }
        }
        bad.unwrap_or(Located::Missing)
    }

    /// Pick the right engine for this GPU and return its local path.
    pub fn resolve(&self, model_id: &str, gpu: &GpuCaps) -> Result<ResolvedEngine> {
        self.check_driver(gpu)?;
        let m = self.find_model(model_id)?;
        let (e, located) = self.pick_engine(m, gpu)?;
        match located {
            Located::Found(path) => Ok(ResolvedEngine {
                id: m.id.clone(),
                kind: m.kind,
                scale: m.scale,
                io_dtype_fp16: m.io_fp16(),
                pad_multiple: m.pad_multiple.max(1),
                path,
                min_shape: e.min_shape,
                opt_shape: e.opt_shape,
                max_shape: e.max_shape,
                arch: e.arch.clone(),
            }),
            Located::BadSize { path, actual } => Err(ModelError::Corrupt {
                id: m.id.clone(),
                path: path.display().to_string(),
                actual,
                expected: e.size,
            }
            .into()),
            Located::Missing => Err(ModelError::NotInstalled {
                id: m.id.clone(),
                file: e.file.clone(),
            }
            .into()),
        }
    }

    /// `(model id, arch, installed)` for every model that has an engine matching this GPU.
    /// Empty if the GPU is unsupported.
    pub fn matching_engines(&self, gpu: &GpuCaps) -> Vec<(String, String, bool)> {
        self.manifest
            .models
            .iter()
            .filter_map(|m| {
                let (e, l) = self.pick_engine(m, gpu).ok()?;
                Some((m.id.clone(), e.arch.clone(), matches!(l, Located::Found(_))))
            })
            .collect()
    }

    /// Every manifest engine file that exists in any search dir (any arch).
    pub fn installed_engines(&self) -> Vec<InstalledEngine> {
        let mut out = Vec::new();
        for m in &self.manifest.models {
            for e in &m.engines {
                for d in &self.dirs {
                    let p = d.join(&e.file);
                    if let Ok(md) = std::fs::metadata(&p) {
                        if md.is_file() {
                            out.push(InstalledEngine {
                                id: m.id.clone(),
                                arch: e.arch.clone(),
                                file: e.file.clone(),
                                path: p,
                                size: md.len(),
                                size_ok: e.size == 0 || md.len() == e.size,
                            });
                        }
                    }
                }
            }
        }
        out
    }

    /// All `manifest.json` files that exist in the search dirs (extra; used by report).
    pub fn manifest_files(&self) -> Vec<PathBuf> {
        let mut v = Vec::new();
        for d in &self.dirs {
            for c in manifest_candidates(d) {
                if c.is_file() && !v.contains(&c) {
                    v.push(c);
                }
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::*;
    use std::fs;

    fn write_engines(dir: &Path, blobs: &[(String, Vec<u8>)]) {
        fs::create_dir_all(dir).unwrap();
        for (n, b) in blobs {
            fs::write(dir.join(n), b).unwrap();
        }
    }

    #[test]
    fn embedded_manifest_is_valid() {
        let m = Manifest::parse(EMBEDDED_MANIFEST).unwrap();
        assert!(m.model("span_x2").is_some());
        assert!(m.model("rife_v4").is_some());
    }

    #[test]
    fn embedded_manifest_keeps_portable_fallback() {
        // every model keeps its portable ampere+ engine; per-cc engines are extra
        let s = ModelStore::with_embedded(vec![]).unwrap();
        for m in &s.manifest().models {
            assert!(m.engine_for_arch("ampere+").is_some(), "{} has no ampere+ engine", m.id);
            for e in &m.engines {
                assert!(e.arch == "ampere+" || e.arch.starts_with("sm"), "{} {}", m.id, e.arch);
            }
            // an unknown future GPU (cc 9.9) still resolves to the portable engine
            assert_eq!(s.select_engine(&m.id, &gpu(9, 9)).unwrap().1.arch, "ampere+");
        }
        let (_, e) = s.select_engine("span_x2", &gpu(8, 6)).unwrap();
        assert_eq!(e.file, "span_x2.sm86.engine");
    }

    #[test]
    fn search_order_and_dedupe() {
        let exe = Path::new("C:/app/bin");
        let v = search_dirs_from(
            Some(PathBuf::from("D:/env")),
            Some(exe),
            Some(PathBuf::from("C:/Users/x/AppData/Local")),
        );
        assert_eq!(v[0], PathBuf::from("D:/env"));
        assert_eq!(v[1], exe.join("models"));
        assert_eq!(
            v[2],
            PathBuf::from("C:/Users/x/AppData/Local/filming-better/models")
        );
        // empty env var is ignored
        let v = search_dirs_from(Some(PathBuf::new()), Some(exe), None);
        assert_eq!(v[0], exe.join("models"));
        // duplicates removed
        let v = search_dirs_from(Some(exe.join("models")), Some(exe), None);
        assert_eq!(v.iter().filter(|d| **d == exe.join("models")).count(), 1);
    }

    #[test]
    fn dev_dir_found_by_walking_up() {
        let t = TempDir::new("dev");
        let eng = t.join("repo/models/engines");
        fs::create_dir_all(&eng).unwrap();
        let exe_dir = t.join("repo/app/target/debug");
        fs::create_dir_all(&exe_dir).unwrap();
        assert_eq!(find_dev_models_dir(&exe_dir), Some(eng.clone()));
        let v = search_dirs_from(None, Some(&exe_dir), None);
        assert_eq!(v.last().unwrap(), &eng);
        // manifest one level above engines/ is found
        fs::write(t.join("repo/models/manifest.json"), sample_manifest("u/").0).unwrap();
        let s = ModelStore::from_dirs(vec![eng.clone()]).unwrap();
        assert_eq!(
            s.manifest_source(),
            &ManifestSource::File(t.join("repo/models/manifest.json"))
        );
    }

    #[test]
    fn manifest_first_dir_wins_and_bad_manifest_skipped() {
        let t = TempDir::new("mf");
        let (a, _) = sample_manifest("http://a/");
        let b = a.replace("http://a/", "http://b/");
        fs::create_dir_all(t.join("d0")).unwrap();
        fs::create_dir_all(t.join("d1")).unwrap();
        fs::create_dir_all(t.join("d2")).unwrap();
        fs::write(t.join("d0/manifest.json"), "{ not json").unwrap();
        fs::write(t.join("d1/manifest.json"), &b).unwrap();
        fs::write(t.join("d2/manifest.json"), &a).unwrap();
        let s = ModelStore::from_dirs(vec![t.join("d0"), t.join("d1"), t.join("d2")]).unwrap();
        assert_eq!(s.manifest().base_url, "http://b/");
        assert_eq!(s.manifest_source(), &ManifestSource::File(t.join("d1/manifest.json")));
        assert_eq!(s.manifest_files().len(), 3);
    }

    #[test]
    fn falls_back_to_embedded() {
        let t = TempDir::new("emb");
        let s = ModelStore::from_dirs(vec![t.join("nothing")]).unwrap();
        assert_eq!(s.manifest_source(), &ManifestSource::Embedded);
        assert_eq!(s.manifest_raw(), EMBEDDED_MANIFEST);
    }

    #[test]
    fn resolve_searches_dirs_in_order() {
        let t = TempDir::new("res");
        let (json, blobs) = sample_manifest("u/");
        write_engines(&t.join("d1"), &blobs);
        write_engines(&t.join("d2"), &blobs);
        let s = ModelStore::with_manifest_text(vec![t.join("d0"), t.join("d1"), t.join("d2")], &json).unwrap();
        let r = s.resolve("span_x2", &gpu(8, 6)).unwrap();
        assert_eq!(r.path, t.join("d1/span_x2.ampere+.engine"));
        assert_eq!(r.id, "span_x2");
        assert_eq!(r.kind, crate::models::ModelKind::Upscale);
        assert_eq!(r.scale, 2);
        assert!(r.io_dtype_fp16);
        assert_eq!(r.arch, "ampere+");
        assert_eq!(r.opt_shape, [1, 3, 540, 960]);
        let r = s.resolve("rife", &gpu(12, 0)).unwrap();
        assert_eq!(r.id, "rife_v4");
        assert_eq!(r.pad_multiple, 64);
        assert!(!r.io_dtype_fp16);
        // sm75 picks the sm75 file
        let r = s.resolve("span_x2", &gpu(7, 5)).unwrap();
        assert_eq!(r.path, t.join("d1/span_x2.sm75.engine"));
        assert_eq!(r.arch, "sm75");
    }

    #[test]
    fn resolve_errors() {
        let t = TempDir::new("err");
        let (json, blobs) = sample_manifest("u/");
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json).unwrap();
        // not installed -> hint
        let e = s.resolve("span_x2", &gpu(8, 6)).unwrap_err();
        assert!(e.to_string().contains("filming-better models pull span_x2"), "{e}");
        assert!(matches!(e.downcast_ref::<ModelError>(), Some(ModelError::NotInstalled { .. })));
        // wrong size -> corrupt, but another dir with the right file still wins
        write_engines(&t.join("d"), &blobs);
        fs::write(t.join("d/span_x2.ampere+.engine"), b"short").unwrap();
        let e = s.resolve("span_x2", &gpu(8, 6)).unwrap_err();
        assert!(matches!(e.downcast_ref::<ModelError>(), Some(ModelError::Corrupt { .. })), "{e}");
        write_engines(&t.join("good"), &blobs);
        let s2 = ModelStore::with_manifest_text(vec![t.join("d"), t.join("good")], &json).unwrap();
        assert_eq!(
            s2.resolve("span_x2", &gpu(8, 6)).unwrap().path,
            t.join("good/span_x2.ampere+.engine")
        );
        // unsupported gpu / unknown model
        let e = s.resolve("span_x2", &gpu(6, 1)).unwrap_err();
        assert!(e.to_string().contains("GPU not supported (needs RTX 20 or newer)"));
        let e = s.resolve("zzz", &gpu(8, 6)).unwrap_err();
        assert!(e.to_string().contains("unknown model"));
        // rife has no sm75 engine in the sample
        let e = s.resolve("rife_v4", &gpu(7, 5)).unwrap_err();
        assert!(matches!(e.downcast_ref::<ModelError>(), Some(ModelError::NoEngineForArch { .. })));
    }

    #[test]
    fn driver_and_trt_checks() {
        let (json, _) = sample_manifest("u/");
        let s = ModelStore::with_manifest_text(vec![], &json).unwrap();
        let mut g = gpu(8, 6);
        g.driver_version = 11040;
        assert!(matches!(s.check_driver(&g), Err(ModelError::DriverTooOld { .. })));
        g.driver_version = 12090;
        assert!(s.check_driver(&g).is_ok());
        g.driver_version = 0; // unknown: do not block
        assert!(s.check_driver(&g).is_ok());
        assert!(s.check_trt_runtime("11.4.0.5").is_ok());
        assert!(s.check_trt_runtime("11").is_ok());
        assert!(matches!(s.check_trt_runtime("10.9.0"), Err(ModelError::TrtMismatch { .. })));
        assert!(s.check_trt_runtime("garbage").is_ok());
    }

    #[test]
    fn size_zero_means_unchecked() {
        let t = TempDir::new("sz0");
        let (json, blobs) = sample_manifest("u/");
        let json0 = json.replace(&format!("\"size\": {}", blobs[0].1.len()), "\"size\": 0");
        write_engines(&t.join("d"), &blobs);
        fs::write(t.join("d/span_x2.ampere+.engine"), b"anything").unwrap();
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json0).unwrap();
        assert!(s.resolve("span_x2", &gpu(8, 6)).is_ok());
    }

    #[test]
    fn matching_and_installed() {
        let t = TempDir::new("match");
        let (json, blobs) = sample_manifest("u/");
        write_engines(&t.join("d"), &blobs[..1]); // only span ampere+
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json).unwrap();
        let m = s.matching_engines(&gpu(8, 6));
        assert_eq!(
            m,
            vec![
                ("span_x2".to_string(), "ampere+".to_string(), true),
                ("rife_v4".to_string(), "ampere+".to_string(), false)
            ]
        );
        let m = s.matching_engines(&gpu(7, 5));
        assert_eq!(m, vec![("span_x2".to_string(), "sm75".to_string(), false)]);
        assert!(s.matching_engines(&gpu(6, 1)).is_empty());
        let inst = s.installed_engines();
        assert_eq!(inst.len(), 1);
        assert_eq!(inst[0].size, 1000);
        assert!(inst[0].size_ok);
    }

    #[test]
    fn sm_engine_preferred_with_ampere_fallback() {
        let t = TempDir::new("sm");
        let (json, mut blobs) = sample_manifest("u/");
        let (json, extra) = add_sm86_engine(&json);
        blobs.push(extra);
        // everything installed
        write_engines(&t.join("all"), &blobs);
        let s = ModelStore::with_manifest_text(vec![t.join("all")], &json).unwrap();
        // cc 8.6: sm86 is picked (manifest lists ampere+ first)
        let r = s.resolve("span_x2", &gpu(8, 6)).unwrap();
        assert_eq!(r.arch, "sm86");
        assert_eq!(r.path, t.join("all/span_x2.sm86.engine"));
        // models without an sm86 engine still use ampere+
        assert_eq!(s.resolve("rife_v4", &gpu(8, 6)).unwrap().arch, "ampere+");
        // cc 8.9 with only sm86 + ampere+ -> ampere+
        assert_eq!(s.resolve("span_x2", &gpu(8, 9)).unwrap().arch, "ampere+");
        assert_eq!(s.resolve("span_x2", &gpu(12, 0)).unwrap().arch, "ampere+");
        // sm75 behaviour unchanged
        assert_eq!(s.resolve("span_x2", &gpu(7, 5)).unwrap().arch, "sm75");
        assert!(matches!(
            s.resolve("rife_v4", &gpu(7, 5)).unwrap_err().downcast_ref::<ModelError>(),
            Some(ModelError::NoEngineForArch { .. })
        ));
        // select_engine (what pull downloads) is manifest-only
        assert_eq!(s.select_engine("span_x2", &gpu(8, 6)).unwrap().1.arch, "sm86");
        assert_eq!(s.select_engine("span_x2", &gpu(8, 9)).unwrap().1.arch, "ampere+");
        assert_eq!(s.select_engine("span_x2", &gpu(7, 5)).unwrap().1.arch, "sm75");
        // matching_engines follows the same order
        assert_eq!(
            s.matching_engines(&gpu(8, 6)),
            vec![
                ("span_x2".to_string(), "sm86".to_string(), true),
                ("rife_v4".to_string(), "ampere+".to_string(), true)
            ]
        );
        assert_eq!(s.matching_engines(&gpu(8, 9))[0].1, "ampere+");

        // portable bundle: only ampere+ files on disk, manifest also lists sm86 -> ampere+ is used
        write_engines(&t.join("portable"), &blobs[..2]);
        let p = ModelStore::with_manifest_text(vec![t.join("portable")], &json).unwrap();
        assert_eq!(p.resolve("span_x2", &gpu(8, 6)).unwrap().arch, "ampere+");
        assert_eq!(p.matching_engines(&gpu(8, 6))[0], ("span_x2".to_string(), "ampere+".to_string(), true));
        // nothing installed: the error asks for pull, and names the best (sm86) file
        let n = ModelStore::with_manifest_text(vec![t.join("empty")], &json).unwrap();
        match n.resolve("span_x2", &gpu(8, 6)).unwrap_err().downcast_ref::<ModelError>() {
            Some(ModelError::NotInstalled { file, .. }) => assert_eq!(file, "span_x2.sm86.engine"),
            o => panic!("{o:?}"),
        }
        assert_eq!(n.matching_engines(&gpu(8, 6))[0], ("span_x2".to_string(), "sm86".to_string(), false));
        // damaged sm86 but valid ampere+ installed -> ampere+; damaged sm86 only -> Corrupt
        write_engines(&t.join("dmg"), &blobs[..2]);
        fs::write(t.join("dmg/span_x2.sm86.engine"), b"x").unwrap();
        let d = ModelStore::with_manifest_text(vec![t.join("dmg")], &json).unwrap();
        assert_eq!(d.resolve("span_x2", &gpu(8, 6)).unwrap().arch, "ampere+");
        fs::remove_file(t.join("dmg/span_x2.ampere+.engine")).unwrap();
        assert!(matches!(
            d.resolve("span_x2", &gpu(8, 6)).unwrap_err().downcast_ref::<ModelError>(),
            Some(ModelError::Corrupt { .. })
        ));
    }
}
