//! Manifest types (schema 1, see docs/contracts.md section 2).
use super::ModelKind;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const SUPPORTED_SCHEMA: u32 = 1;

fn one() -> u32 {
    1
}

fn default_dtype() -> String {
    "fp16".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub trt_version: String,
    pub cuda_major: u32,
    pub base_url: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub kind: ModelKind,
    #[serde(default = "one")]
    pub scale: u32,
    #[serde(default = "default_dtype")]
    pub io_dtype: String,
    #[serde(default = "one")]
    pub pad_multiple: u32,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub engines: Vec<EngineEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineEntry {
    /// `"ampere+"` (portable, cc >= 8.0), `"sm<major><minor>"` (exact compute capability,
    /// e.g. `sm75`, `sm86`, `sm120`) or a future tag (never selected by this build).
    pub arch: String,
    pub file: String,
    pub sha256: String,
    pub size: u64,
    pub min_shape: [i64; 4],
    pub opt_shape: [i64; 4],
    pub max_shape: [i64; 4],
    #[serde(default)]
    pub fps_1080p: Option<f64>,
    #[serde(default)]
    pub build_gpu: Option<String>,
}

impl ModelEntry {
    pub fn io_fp16(&self) -> bool {
        !self.io_dtype.eq_ignore_ascii_case("fp32")
    }
    pub fn engine_for_arch(&self, arch: &str) -> Option<&EngineEntry> {
        self.engines.iter().find(|e| e.arch == arch)
    }
}

impl Manifest {
    /// Parse and validate. Unknown fields are ignored (forward compatibility).
    pub fn parse(text: &str) -> Result<Manifest> {
        // Tolerate a UTF-8 BOM (Windows editors).
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let m: Manifest = serde_json::from_str(text).context("invalid manifest.json")?;
        m.validate()?;
        Ok(m)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != SUPPORTED_SCHEMA {
            bail!(
                "unsupported manifest schema {} (this build understands schema {}); update filming-better",
                self.schema,
                SUPPORTED_SCHEMA
            );
        }
        for m in &self.models {
            if m.id.is_empty() {
                bail!("manifest: model with empty id");
            }
            for e in &m.engines {
                if !is_safe_file_name(&e.file) {
                    bail!(
                        "manifest: model '{}' has an unsafe engine file name '{}'",
                        m.id,
                        e.file
                    );
                }
            }
        }
        Ok(())
    }

    pub fn model(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// `base_url` (or the override) + `file`, with exactly one slash between.
    pub fn engine_url(&self, base_override: Option<&str>, e: &EngineEntry) -> String {
        let base = base_override.unwrap_or(&self.base_url);
        if base.ends_with('/') {
            format!("{}{}", base, e.file)
        } else {
            format!("{}/{}", base, e.file)
        }
    }

    /// TensorRT major version the engines were built with ("11.4.0" -> 11).
    pub fn trt_major(&self) -> Option<u32> {
        self.trt_version.split('.').next()?.trim().parse().ok()
    }
}

/// A plain file name: no separators, no drive prefix, not "." / "..", not hidden.
pub fn is_safe_file_name(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('.') && !s.contains(['/', '\\', ':', '\0'])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::sample_manifest;

    #[test]
    fn parses_sample_and_ignores_unknown_fields() {
        let (json, _) = sample_manifest("http://x/");
        let m = Manifest::parse(&json).unwrap();
        assert_eq!(m.schema, 1);
        assert_eq!(m.models.len(), 2);
        let s = m.model("span_x2").unwrap();
        assert_eq!(s.kind, ModelKind::Upscale);
        assert_eq!(s.scale, 2);
        assert!(s.io_fp16());
        assert_eq!(s.engines.len(), 2);
        assert_eq!(s.engines[0].opt_shape, [1, 3, 540, 960]);
        assert_eq!(s.engines[0].fps_1080p, Some(41.2));
        assert_eq!(s.engines[1].fps_1080p, None);
        let r = m.model("rife_v4").unwrap();
        assert_eq!(r.kind, ModelKind::Interp);
        assert_eq!(r.pad_multiple, 64);
        assert!(!r.io_fp16());
        assert_eq!(m.trt_major(), Some(11));
    }

    #[test]
    fn rejects_other_schema() {
        let e = Manifest::parse(
            r#"{"schema":2,"trt_version":"11","cuda_major":12,"base_url":"x","models":[]}"#,
        )
        .unwrap_err();
        assert!(e.to_string().contains("schema 2"), "{e}");
    }

    #[test]
    fn rejects_garbage_and_missing_fields() {
        assert!(Manifest::parse("nope").is_err());
        assert!(Manifest::parse(r#"{"schema":1}"#).is_err());
    }

    #[test]
    fn rejects_path_traversal_file_names() {
        let (json, _) = sample_manifest("x");
        let bad = json.replace("span_x2.sm75.engine", "..\\\\evil.engine");
        assert!(Manifest::parse(&bad).is_err());
        assert!(!is_safe_file_name("a/b"));
        assert!(!is_safe_file_name("C:x"));
        assert!(!is_safe_file_name(".."));
        assert!(is_safe_file_name("span_x2.ampere+.engine"));
    }

    #[test]
    fn bom_tolerated_and_url_join() {
        let (json, _) = sample_manifest("http://h/p");
        let m = Manifest::parse(&format!("\u{feff}{json}")).unwrap();
        let e = &m.models[0].engines[0];
        assert_eq!(m.engine_url(None, e), "http://h/p/span_x2.ampere+.engine");
        assert_eq!(
            m.engine_url(Some("http://q/"), e),
            "http://q/span_x2.ampere+.engine"
        );
    }

    #[test]
    fn defaults_for_optional_fields() {
        let m = Manifest::parse(
            r#"{"schema":1,"trt_version":"11.4.0","cuda_major":12,"base_url":"u/",
            "models":[{"id":"a","kind":"interp","engines":[]}]}"#,
        )
        .unwrap();
        assert_eq!(m.models[0].pad_multiple, 1);
        assert_eq!(m.models[0].io_dtype, "fp16");
    }
}
