//! Engine architecture selection by compute capability, and model-id matching.
use super::manifest::{EngineEntry, ModelEntry};
use super::{GpuCaps, ModelError};

pub const ARCH_AMPERE_PLUS: &str = "ampere+";
pub const ARCH_SM75: &str = "sm75";

/// Exact per-compute-capability tag: cc 8.6 -> "sm86", cc 12.0 -> "sm120".
pub fn sm_tag(gpu: &GpuCaps) -> String {
    format!("sm{}{}", gpu.cc_major, gpu.cc_minor)
}

fn unsupported(gpu: &GpuCaps) -> ModelError {
    ModelError::UnsupportedGpu {
        name: gpu.name.clone(),
        major: gpu.cc_major,
        minor: gpu.cc_minor,
    }
}

/// Generic (fallback) arch: cc >= 8.0 -> "ampere+", cc == 7.5 -> "sm75", otherwise unsupported.
pub fn engine_arch(gpu: &GpuCaps) -> Result<&'static str, ModelError> {
    let cc = (gpu.cc_major, gpu.cc_minor);
    if cc >= (8, 0) {
        Ok(ARCH_AMPERE_PLUS)
    } else if cc == (7, 5) {
        Ok(ARCH_SM75)
    } else {
        Err(unsupported(gpu))
    }
}

/// Engine arch tags acceptable for this GPU, best first:
/// the exact `sm<major><minor>` tag, then the generic fallback (`ampere+` for cc >= 8.0;
/// for cc 7.5 the exact tag is already `sm75`). Unsupported GPUs are an error.
pub fn arch_candidates(gpu: &GpuCaps) -> Result<Vec<String>, ModelError> {
    let generic = engine_arch(gpu)?;
    let exact = sm_tag(gpu);
    if exact == generic {
        Ok(vec![exact])
    } else {
        Ok(vec![exact, generic.to_string()])
    }
}

/// Engines of `m` usable on `gpu`, best first (manifest order is irrelevant).
pub fn ranked_engines<'a>(
    m: &'a ModelEntry,
    gpu: &GpuCaps,
) -> Result<Vec<&'a EngineEntry>, ModelError> {
    let cands = arch_candidates(gpu)?;
    Ok(cands.iter().filter_map(|a| m.engine_for_arch(a)).collect())
}

/// The best engine of `m` for `gpu` according to the manifest alone (installed or not).
pub fn best_engine<'a>(m: &'a ModelEntry, gpu: &GpuCaps) -> Result<&'a EngineEntry, ModelError> {
    ranked_engines(m, gpu)?
        .into_iter()
        .next()
        .ok_or_else(|| ModelError::NoEngineForArch {
            id: m.id.clone(),
            arch: arch_candidates(gpu).map(|c| c.join("/")).unwrap_or_default(),
        })
}

/// Short aliases accepted in addition to unique-prefix matching.
const ALIASES: &[(&str, &str)] = &[
    ("span", "span_x2"),
    ("span2x", "span_x2"),
    ("anime", "animevideov3_x4"),
    ("animevideo", "animevideov3_x4"),
    ("animevideov3", "animevideov3_x4"),
    ("rife", "rife_v4"),
];

fn norm(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace('-', "_")
}

/// Resolve a user-supplied model id against the available ids.
/// Order: exact, normalized-exact, alias, unique prefix.
pub fn match_model_id<'a>(input: &str, ids: &[&'a str]) -> Result<&'a str, ModelError> {
    if let Some(id) = ids.iter().find(|i| **i == input) {
        return Ok(id);
    }
    let n = norm(input);
    if let Some(id) = ids.iter().find(|i| norm(i) == n) {
        return Ok(id);
    }
    if let Some((_, target)) = ALIASES.iter().find(|(a, _)| *a == n) {
        if let Some(id) = ids.iter().find(|i| **i == *target) {
            return Ok(id);
        }
    }
    if !n.is_empty() {
        let m: Vec<&str> = ids.iter().copied().filter(|i| norm(i).starts_with(&n)).collect();
        match m.len() {
            1 => return Ok(m[0]),
            0 => {}
            _ => {
                return Err(ModelError::AmbiguousModel {
                    id: input.to_string(),
                    matches: m.join(", "),
                })
            }
        }
    }
    Err(ModelError::UnknownModel {
        id: input.to_string(),
        available: ids.join(", "),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::gpu;

    #[test]
    fn arch_by_cc() {
        assert_eq!(engine_arch(&gpu(8, 6)).unwrap(), "ampere+");
        assert_eq!(engine_arch(&gpu(8, 0)).unwrap(), "ampere+");
        assert_eq!(engine_arch(&gpu(8, 9)).unwrap(), "ampere+");
        assert_eq!(engine_arch(&gpu(12, 0)).unwrap(), "ampere+");
        assert_eq!(engine_arch(&gpu(7, 5)).unwrap(), "sm75");
    }

    #[test]
    fn candidates_exact_first() {
        assert_eq!(arch_candidates(&gpu(8, 6)).unwrap(), ["sm86", "ampere+"]);
        assert_eq!(arch_candidates(&gpu(8, 9)).unwrap(), ["sm89", "ampere+"]);
        assert_eq!(arch_candidates(&gpu(12, 0)).unwrap(), ["sm120", "ampere+"]);
        assert_eq!(arch_candidates(&gpu(7, 5)).unwrap(), ["sm75"]);
        assert!(arch_candidates(&gpu(7, 0)).is_err());
    }

    fn model_with(archs: &[&str]) -> ModelEntry {
        let e = |a: &str| EngineEntry {
            arch: a.to_string(),
            file: format!("m.{a}.engine"),
            sha256: String::new(),
            size: 1,
            min_shape: [1, 3, 64, 64],
            opt_shape: [1, 3, 64, 64],
            max_shape: [1, 3, 64, 64],
            fps_1080p: None,
            build_gpu: None,
        };
        ModelEntry {
            id: "m".into(),
            kind: crate::models::ModelKind::Upscale,
            scale: 1,
            io_dtype: "fp16".into(),
            pad_multiple: 1,
            description: String::new(),
            license: String::new(),
            source: String::new(),
            engines: archs.iter().map(|a| e(a)).collect(),
        }
    }

    #[test]
    fn best_engine_order() {
        // exact tag wins regardless of manifest order
        let m = model_with(&["ampere+", "sm86"]);
        assert_eq!(best_engine(&m, &gpu(8, 6)).unwrap().arch, "sm86");
        // other cc >= 8.0 falls back to ampere+, not to a foreign sm tag
        let m = model_with(&["sm86", "ampere+"]);
        assert_eq!(best_engine(&m, &gpu(8, 9)).unwrap().arch, "ampere+");
        assert_eq!(best_engine(&m, &gpu(8, 0)).unwrap().arch, "ampere+");
        // only a foreign sm tag -> none
        let m = model_with(&["sm86"]);
        assert!(matches!(
            best_engine(&m, &gpu(8, 9)),
            Err(ModelError::NoEngineForArch { .. })
        ));
        // sm75 unchanged: ampere+ is never used on Turing
        let m = model_with(&["ampere+", "sm86", "sm75"]);
        assert_eq!(best_engine(&m, &gpu(7, 5)).unwrap().arch, "sm75");
        let m = model_with(&["ampere+", "sm86"]);
        assert!(matches!(
            best_engine(&m, &gpu(7, 5)),
            Err(ModelError::NoEngineForArch { .. })
        ));
        assert_eq!(ranked_engines(&model_with(&["ampere+", "sm86"]), &gpu(8, 6)).unwrap().len(), 2);
    }

    #[test]
    fn unsupported_gpus() {
        for (a, b) in [(6, 1), (7, 0), (7, 2), (5, 2), (0, 0)] {
            let e = engine_arch(&gpu(a, b)).unwrap_err();
            let s = e.to_string();
            assert!(
                s.starts_with("GPU not supported (needs RTX 20 or newer)"),
                "{s}"
            );
        }
    }

    #[test]
    fn model_id_matching() {
        let ids = ["span_x2", "animevideov3_x4", "rife_v4"];
        assert_eq!(match_model_id("span_x2", &ids).unwrap(), "span_x2");
        assert_eq!(match_model_id("SPAN", &ids).unwrap(), "span_x2");
        assert_eq!(match_model_id("rife", &ids).unwrap(), "rife_v4");
        assert_eq!(match_model_id("rife-v4", &ids).unwrap(), "rife_v4");
        assert_eq!(match_model_id("animevideo", &ids).unwrap(), "animevideov3_x4");
        assert_eq!(match_model_id("anim", &ids).unwrap(), "animevideov3_x4");
        assert!(matches!(
            match_model_id("nope", &ids),
            Err(ModelError::UnknownModel { .. })
        ));
        assert!(matches!(
            match_model_id("", &ids),
            Err(ModelError::UnknownModel { .. })
        ));
        let ids2 = ["span_x2", "span_x4"];
        assert!(matches!(
            match_model_id("sp", &ids2),
            Err(ModelError::AmbiguousModel { .. })
        ));
        assert_eq!(match_model_id("span_x4", &ids2).unwrap(), "span_x4");
    }
}
