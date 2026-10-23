//! TOML config file + presets + defaults merging.
//!
//! Precedence for run options: explicit CLI value > preset > config file > built-in default.
//! (Exception: when the preset itself is named only in the config file, values written
//! explicitly in that same file win over the preset, so `preset = "fast"` + `cq = 18` means cq 18.)

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub const DEFAULT_INTERP: u32 = 2;
pub const DEFAULT_CODEC: &str = "hevc";
pub const DEFAULT_CQ: u32 = 20;

/// Names accepted by `--preset`.
pub const PRESET_NAMES: [&str; 3] = ["fast", "balanced", "quality"];

/// Keys understood in the config file (anything else only produces a warning).
const KNOWN_KEYS: [&str; 11] = [
    "preset",
    "upscale",
    "interp",
    "codec",
    "cq",
    "order",
    "scene_threshold",
    "output_dir",
    "name_template",
    "tile",
    "log_file",
];

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RunDefaults {
    pub preset: Option<String>,
    pub upscale: Option<String>,
    pub interp: Option<u32>,
    pub codec: Option<String>,
    pub cq: Option<u32>,
    pub order: Option<String>,
    pub scene_threshold: Option<f32>,
    pub output_dir: Option<PathBuf>,
    pub name_template: Option<String>,
    pub tile: Option<u32>,
    pub log_file: Option<PathBuf>,
}

/// Default config path: `%APPDATA%\filming-better\config.toml`.
pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("filming-better").join("config.toml"))
}

/// `--config <path>` (must exist) or the default file (missing = built-in defaults).
pub fn load_config(explicit: Option<&Path>) -> Result<RunDefaults> {
    match explicit {
        Some(p) => load_file(p, true),
        None => match default_config_path() {
            Some(p) => load_file(&p, false),
            None => Ok(RunDefaults::default()),
        },
    }
}

/// Load one file. `must_exist = false`: a missing file gives defaults.
pub fn load_file(path: &Path, must_exist: bool) -> Result<RunDefaults> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !must_exist => {
            return Ok(RunDefaults::default())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("config file not found: {}", path.display())
        }
        Err(e) => return Err(e).with_context(|| format!("cannot read config file {}", path.display())),
    };
    let (cfg, warnings) =
        parse_config(&text).with_context(|| format!("invalid config file {}", path.display()))?;
    for w in warnings {
        tracing::warn!("{}: {}", path.display(), w);
        eprintln!("warning: {}: {}", path.display(), w);
    }
    Ok(cfg)
}

/// Parse + validate TOML text; returns warnings for unknown keys.
pub fn parse_config(text: &str) -> Result<(RunDefaults, Vec<String>)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let table: toml::Table = text.parse().context("TOML syntax error")?;
    let warnings: Vec<String> = table
        .keys()
        .filter(|k| !KNOWN_KEYS.contains(&k.as_str()))
        .map(|k| format!("unknown option `{}` ignored", k))
        .collect();
    let cfg: RunDefaults = table.try_into().context("wrong option type or value")?;
    cfg.validate()?;
    Ok((cfg, warnings))
}

impl RunDefaults {
    pub fn validate(&self) -> Result<()> {
        if let Some(p) = &self.preset {
            if preset(p).is_none() {
                bail!("unknown preset '{}' (use one of: {})", p, PRESET_NAMES.join(", "));
            }
        }
        if let Some(c) = &self.codec {
            if !["h264", "hevc", "av1"].contains(&c.to_ascii_lowercase().as_str()) {
                bail!("unknown codec '{}' (use h264, hevc or av1)", c);
            }
        }
        if let Some(i) = self.interp {
            if i == 0 || i > 8 {
                bail!("interp must be between 1 and 8 (got {})", i);
            }
        }
        if let Some(q) = self.cq {
            if q > 51 {
                bail!("cq must be between 0 and 51 (got {})", q);
            }
        }
        if let Some(t) = self.scene_threshold {
            if !t.is_finite() || t < 0.0 {
                bail!("scene_threshold must be a non-negative number (got {})", t);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Preset {
    pub upscale: Option<&'static str>,
    pub interp: u32,
    pub codec: &'static str,
    pub cq: u32,
}

/// `fast | balanced | quality` (case-insensitive).
pub fn preset(name: &str) -> Option<Preset> {
    match name.trim().to_ascii_lowercase().as_str() {
        "fast" => Some(Preset { upscale: Some("span_x2"), interp: 2, codec: "hevc", cq: 24 }),
        "balanced" => Some(Preset { upscale: Some("span_x2"), interp: 2, codec: "hevc", cq: 20 }),
        "quality" => Some(Preset { upscale: Some("animevideov3_x4"), interp: 2, codec: "hevc", cq: 18 }),
        _ => None,
    }
}

/// First `Some` of cli, preset, config; else `default`.
pub fn merge<T>(cli: Option<T>, preset: Option<T>, config: Option<T>, default: T) -> T {
    cli.or(preset).or(config).unwrap_or(default)
}

/// Like [`merge`] with no built-in default.
pub fn merge_opt<T>(cli: Option<T>, preset: Option<T>, config: Option<T>) -> Option<T> {
    cli.or(preset).or(config)
}

/// Merge CLI-provided options (`cli`, all `None` when not given) with the config file.
/// The result has `interp`, `codec`, `cq` always set; `upscale` is set if any layer gives one.
pub fn resolve_run(cli: &RunDefaults, config: &RunDefaults) -> Result<RunDefaults> {
    let (name, from_cli) = match (&cli.preset, &config.preset) {
        (Some(p), _) => (Some(p.clone()), true),
        (None, Some(p)) => (Some(p.clone()), false),
        _ => (None, false),
    };
    let p = match &name {
        Some(n) => Some(preset(n).with_context(|| {
            format!("unknown preset '{}' (use one of: {})", n, PRESET_NAMES.join(", "))
        })?),
        None => None,
    };
    // Layer order of (preset, config): CLI-chosen preset beats the config file;
    // a preset named only in the config file is overridden by that file's own values.
    fn pick<T>(cli: Option<T>, preset: Option<T>, config: Option<T>, from_cli: bool) -> Option<T> {
        if from_cli {
            merge_opt(cli, preset, config)
        } else {
            merge_opt(cli, config, preset)
        }
    }
    Ok(RunDefaults {
        preset: name,
        upscale: pick(
            cli.upscale.clone(),
            p.and_then(|p| p.upscale).map(String::from),
            config.upscale.clone(),
            from_cli,
        ),
        interp: Some(
            pick(cli.interp, p.map(|p| p.interp), config.interp, from_cli).unwrap_or(DEFAULT_INTERP),
        ),
        codec: Some(
            pick(
                cli.codec.clone(),
                p.map(|p| p.codec.to_string()),
                config.codec.clone(),
                from_cli,
            )
            .unwrap_or_else(|| DEFAULT_CODEC.to_string()),
        ),
        cq: Some(pick(cli.cq, p.map(|p| p.cq), config.cq, from_cli).unwrap_or(DEFAULT_CQ)),
        order: cli.order.clone().or_else(|| config.order.clone()),
        scene_threshold: cli.scene_threshold.or(config.scene_threshold),
        output_dir: cli.output_dir.clone().or_else(|| config.output_dir.clone()),
        name_template: cli.name_template.clone().or_else(|| config.name_template.clone()),
        tile: cli.tile.or(config.tile),
        log_file: cli.log_file.clone().or_else(|| config.log_file.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::TempDir;

    #[test]
    fn preset_values() {
        let f = preset("fast").unwrap();
        assert_eq!((f.upscale, f.interp, f.codec, f.cq), (Some("span_x2"), 2, "hevc", 24));
        let b = preset("Balanced").unwrap();
        assert_eq!((b.upscale, b.interp, b.codec, b.cq), (Some("span_x2"), 2, "hevc", 20));
        let q = preset("QUALITY").unwrap();
        assert_eq!((q.upscale, q.interp, q.codec, q.cq), (Some("animevideov3_x4"), 2, "hevc", 18));
        assert!(preset("ultra").is_none());
        for n in PRESET_NAMES {
            assert!(preset(n).is_some());
        }
    }

    #[test]
    fn merge_precedence() {
        assert_eq!(merge(Some(1), Some(2), Some(3), 4), 1);
        assert_eq!(merge(None, Some(2), Some(3), 4), 2);
        assert_eq!(merge(None, None, Some(3), 4), 3);
        assert_eq!(merge::<i32>(None, None, None, 4), 4);
        assert_eq!(merge_opt::<i32>(None, None, None), None);
        assert_eq!(merge_opt(None, None, Some("c")), Some("c"));
    }

    #[test]
    fn parse_full_config() {
        let (c, w) = parse_config(
            r#"
preset = "balanced"
upscale = "span_x2"
interp = 3
codec = "av1"
cq = 22
order = "interp-first"
scene_threshold = 0.35
output_dir = "D:\\out"
name_template = "{stem}_fb"
tile = 512
log_file = "C:\\logs\\fb.log"
"#,
        )
        .unwrap();
        assert!(w.is_empty());
        assert_eq!(c.preset.as_deref(), Some("balanced"));
        assert_eq!(c.interp, Some(3));
        assert_eq!(c.cq, Some(22));
        assert_eq!(c.scene_threshold, Some(0.35));
        assert_eq!(c.output_dir, Some(PathBuf::from("D:\\out")));
        assert_eq!(c.tile, Some(512));
    }

    #[test]
    fn parse_errors_and_unknown_keys() {
        let (c, w) = parse_config("upscle = \"x\"\ncq = 20\n").unwrap();
        assert_eq!(c.cq, Some(20));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("upscle"));
        assert!(parse_config("cq = \"high\"").is_err());
        assert!(parse_config("cq = ").is_err());
        assert!(parse_config("codec = \"vp9\"").unwrap_err().to_string().contains("codec"));
        assert!(parse_config("preset = \"x\"").is_err());
        assert!(parse_config("interp = 0").is_err());
        assert!(parse_config("cq = 99").is_err());
        assert!(parse_config("scene_threshold = -1.0").is_err());
        assert_eq!(parse_config("").unwrap().0, RunDefaults::default());
    }

    #[test]
    fn explicit_missing_file_is_error_default_missing_is_ok() {
        let t = TempDir::new("cfg");
        let missing = t.join("nope.toml");
        let e = load_config(Some(&missing)).unwrap_err();
        assert!(e.to_string().contains("config file not found"), "{e}");
        assert_eq!(load_file(&missing, false).unwrap(), RunDefaults::default());
        // present file loads
        std::fs::write(t.join("c.toml"), "cq = 19\n").unwrap();
        assert_eq!(load_config(Some(&t.join("c.toml"))).unwrap().cq, Some(19));
        // invalid file error mentions the path
        std::fs::write(t.join("bad.toml"), "cq = [").unwrap();
        let e = load_config(Some(&t.join("bad.toml"))).unwrap_err();
        assert!(format!("{e:#}").contains("bad.toml"), "{e:#}");
        // a directory is not a file
        assert!(load_file(t.path(), true).is_err());
    }

    #[test]
    fn default_path_shape() {
        if let Some(p) = default_config_path() {
            assert!(p.ends_with("filming-better/config.toml") || p.ends_with("filming-better\\config.toml"));
        }
    }

    #[test]
    fn resolve_builtin_defaults() {
        let r = resolve_run(&RunDefaults::default(), &RunDefaults::default()).unwrap();
        assert_eq!((r.interp, r.codec.as_deref(), r.cq), (Some(2), Some("hevc"), Some(20)));
        assert_eq!(r.upscale, None);
        assert_eq!(r.preset, None);
    }

    #[test]
    fn resolve_cli_preset_beats_config() {
        let cli = RunDefaults { preset: Some("quality".into()), ..Default::default() };
        let cfg = RunDefaults {
            upscale: Some("span_x2".into()),
            cq: Some(30),
            interp: Some(4),
            codec: Some("h264".into()),
            tile: Some(256),
            ..Default::default()
        };
        let r = resolve_run(&cli, &cfg).unwrap();
        assert_eq!(r.upscale.as_deref(), Some("animevideov3_x4"));
        assert_eq!(r.cq, Some(18));
        assert_eq!(r.interp, Some(2));
        assert_eq!(r.codec.as_deref(), Some("hevc"));
        assert_eq!(r.tile, Some(256)); // not covered by presets: config applies
    }

    #[test]
    fn resolve_explicit_cli_beats_preset() {
        let cli = RunDefaults {
            preset: Some("fast".into()),
            cq: Some(15),
            upscale: Some("animevideov3_x4".into()),
            tile: Some(128),
            ..Default::default()
        };
        let cfg = RunDefaults { tile: Some(256), cq: Some(30), ..Default::default() };
        let r = resolve_run(&cli, &cfg).unwrap();
        assert_eq!(r.cq, Some(15));
        assert_eq!(r.upscale.as_deref(), Some("animevideov3_x4"));
        assert_eq!(r.interp, Some(2)); // from preset
        assert_eq!(r.tile, Some(128));
    }

    #[test]
    fn resolve_config_preset_is_overridden_by_config_values() {
        let cfg = RunDefaults { preset: Some("fast".into()), cq: Some(18), ..Default::default() };
        let r = resolve_run(&RunDefaults::default(), &cfg).unwrap();
        assert_eq!(r.cq, Some(18));
        assert_eq!(r.upscale.as_deref(), Some("span_x2"));
        assert_eq!(r.preset.as_deref(), Some("fast"));
        // CLI still wins over everything
        let cli = RunDefaults { cq: Some(10), ..Default::default() };
        assert_eq!(resolve_run(&cli, &cfg).unwrap().cq, Some(10));
        // CLI preset replaces the config preset
        let cli = RunDefaults { preset: Some("balanced".into()), ..Default::default() };
        let r = resolve_run(&cli, &cfg).unwrap();
        assert_eq!(r.cq, Some(20));
    }

    #[test]
    fn resolve_unknown_preset_errors() {
        let cli = RunDefaults { preset: Some("turbo".into()), ..Default::default() };
        let e = resolve_run(&cli, &RunDefaults::default()).unwrap_err();
        assert!(e.to_string().contains("unknown preset 'turbo'"), "{e}");
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../docs/config.example.toml");
        let (_, warnings) = parse_config(text).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }
}
