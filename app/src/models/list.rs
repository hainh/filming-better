//! `models list`: table of models.
use super::select::engine_arch;
use super::store::{Located, ModelStore};
use super::GpuCaps;
use anyhow::Result;

pub fn human_size(n: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    if n == 0 {
        "?".into()
    } else if (n as f64) >= 1024.0 * MB {
        format!("{:.2} GB", n as f64 / (1024.0 * MB))
    } else if (n as f64) >= MB {
        format!("{:.1} MB", n as f64 / MB)
    } else {
        format!("{:.1} KB", n as f64 / 1024.0)
    }
}

/// Render the table. With a GPU, one row per model for the engine that matches it;
/// without, one row per engine (all archs).
pub fn render_list(store: &ModelStore, gpu: Option<&GpuCaps>) -> String {
    let header = ["ID", "KIND", "SCALE", "ARCH", "INSTALLED", "SIZE", "FPS@1080P", "DESCRIPTION"];
    let mut rows: Vec<[String; 8]> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    // Err = unsupported GPU; Ok = generic arch name shown for models without an engine.
    let arch_for_gpu = gpu.map(engine_arch);
    if let Some(Err(e)) = &arch_for_gpu {
        notes.push(e.to_string());
    }

    for m in &store.manifest().models {
        let engines: Vec<Option<(&super::manifest::EngineEntry, Located)>> = match (&arch_for_gpu, gpu) {
            // Same pick as `resolve`: exact sm<cc>, else ampere+ / sm75 (installed beats better-ranked missing).
            (Some(Ok(_)), Some(g)) => vec![store.pick_engine(m, g).ok()],
            (Some(Err(_)), _) | (Some(Ok(_)), None) => vec![None],
            (None, _) => {
                if m.engines.is_empty() {
                    vec![None]
                } else {
                    m.engines.iter().map(|e| Some((e, store.locate(e)))).collect()
                }
            }
        };
        for e in engines {
            let (arch, inst, size, fps) = match e {
                Some((e, located)) => (
                    e.arch.clone(),
                    match located {
                        Located::Found(_) => "yes",
                        Located::BadSize { .. } => "damaged",
                        Located::Missing => "no",
                    }
                    .to_string(),
                    human_size(e.size),
                    e.fps_1080p.map(|f| format!("{:.1}", f)).unwrap_or_else(|| "-".into()),
                ),
                None => {
                    let a = match &arch_for_gpu {
                        Some(Ok(a)) => format!("{} (n/a)", a),
                        _ => "-".to_string(),
                    };
                    (a, "-".into(), "-".into(), "-".into())
                }
            };
            rows.push([
                m.id.clone(),
                m.kind.to_string(),
                if m.scale > 1 { format!("x{}", m.scale) } else { "-".into() },
                arch,
                inst,
                size,
                fps,
                m.description.clone(),
            ]);
        }
    }

    let mut w: Vec<usize> = header.iter().map(|h| h.len()).collect();
    for r in &rows {
        for (i, c) in r.iter().enumerate() {
            w[i] = w[i].max(c.chars().count());
        }
    }
    let line = |cells: Vec<&str>| -> String {
        let mut s = String::new();
        for (i, c) in cells.iter().enumerate() {
            if i + 1 == cells.len() {
                s.push_str(c);
            } else {
                s.push_str(&format!("{:<width$}  ", c, width = w[i]));
            }
        }
        s.trim_end().to_string()
    };
    let mut out = String::new();
    out.push_str(&line(header.to_vec()));
    out.push('\n');
    for r in &rows {
        out.push_str(&line(r.iter().map(|s| s.as_str()).collect()));
        out.push('\n');
    }
    if let Some(g) = gpu {
        out.push_str(&format!("\nGPU: {} (compute capability {}.{})\n", g.name, g.cc_major, g.cc_minor));
    }
    for n in notes {
        out.push_str(&format!("Note: {}\n", n));
    }
    out.push_str(&format!(
        "Manifest: {}\nSearch dirs: {}\n",
        match store.manifest_source() {
            super::store::ManifestSource::Embedded => "built-in".to_string(),
            super::store::ManifestSource::File(p) => p.display().to_string(),
        },
        store
            .dirs()
            .iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join("; ")
    ));
    out
}

pub fn cmd_list(gpu: Option<&GpuCaps>) -> Result<()> {
    let store = ModelStore::open()?;
    print!("{}", render_list(&store, gpu));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::testutil::*;
    use std::fs;

    #[test]
    fn human_sizes() {
        assert_eq!(human_size(0), "?");
        assert_eq!(human_size(2048), "2.0 KB");
        assert_eq!(human_size(5 * 1024 * 1024), "5.0 MB");
        assert_eq!(human_size(3 * 1024 * 1024 * 1024), "3.00 GB");
    }

    #[test]
    fn list_with_and_without_gpu() {
        let t = TempDir::new("list");
        let (json, blobs) = sample_manifest("u/");
        fs::create_dir_all(t.join("d")).unwrap();
        fs::write(t.join("d").join(&blobs[0].0), &blobs[0].1).unwrap();
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json).unwrap();

        let txt = render_list(&s, Some(&gpu(8, 6)));
        let span = txt.lines().find(|l| l.starts_with("span_x2")).unwrap();
        assert!(span.contains("x2") && span.contains("ampere+") && span.contains("yes"), "{span}");
        assert!(span.contains("41.2"), "{span}");
        let rife = txt.lines().find(|l| l.starts_with("rife_v4")).unwrap();
        assert!(rife.contains("no") && rife.contains("interp"), "{rife}");
        assert!(txt.contains("GPU: Test GPU 8.6"));

        let txt = render_list(&s, Some(&gpu(7, 5)));
        let rife = txt.lines().find(|l| l.starts_with("rife_v4")).unwrap();
        assert!(rife.contains("sm75 (n/a)"), "{rife}");

        let txt = render_list(&s, Some(&gpu(6, 1)));
        assert!(txt.contains("GPU not supported"), "{txt}");

        let txt = render_list(&s, None);
        assert_eq!(txt.lines().filter(|l| l.starts_with("span_x2")).count(), 2);
        assert!(txt.starts_with("ID"));
    }

    #[test]
    fn list_prefers_exact_sm_engine() {
        let t = TempDir::new("list-sm");
        let (json, mut blobs) = sample_manifest("u/");
        let (json, extra) = add_sm86_engine(&json);
        blobs.push(extra);
        fs::create_dir_all(t.join("d")).unwrap();
        // only the ampere+ span engine is installed
        fs::write(t.join("d").join(&blobs[0].0), &blobs[0].1).unwrap();
        let s = ModelStore::with_manifest_text(vec![t.join("d")], &json).unwrap();
        // cc 8.6: sm86 is listed but not installed; the installed ampere+ one is used (same as resolve)
        let txt = render_list(&s, Some(&gpu(8, 6)));
        let span = txt.lines().find(|l| l.starts_with("span_x2")).unwrap();
        assert!(span.contains("ampere+") && span.contains("yes"), "{span}");
        // install sm86 too -> it wins
        fs::write(t.join("d").join(&blobs[3].0), &blobs[3].1).unwrap();
        let txt = render_list(&s, Some(&gpu(8, 6)));
        let span = txt.lines().find(|l| l.starts_with("span_x2")).unwrap();
        assert!(span.contains("sm86") && span.contains("yes"), "{span}");
        // cc 8.9: foreign sm86 is ignored
        let txt = render_list(&s, Some(&gpu(8, 9)));
        let span = txt.lines().find(|l| l.starts_with("span_x2")).unwrap();
        assert!(span.contains("ampere+") && !span.contains("sm86"), "{span}");
        // nothing installed: shows the best (sm86) as not installed
        let s2 = ModelStore::with_manifest_text(vec![t.join("none")], &json).unwrap();
        let txt = render_list(&s2, Some(&gpu(8, 6)));
        let span = txt.lines().find(|l| l.starts_with("span_x2")).unwrap();
        assert!(span.contains("sm86") && span.contains("no"), "{span}");
    }
}
