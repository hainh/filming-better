//! Build script: builds `native/` (fb_native.dll) with CMake+Ninja, generates the Rust
//! bindings from `native/include/fb_native.h`, links the import library and copies the
//! runtime DLLs next to the produced executable.
//!
//! Environment knobs:
//!   FB_SKIP_NATIVE=1        skip CMake + linking (only bindgen) - for `cargo check` while
//!                           native/ is not ready.
//!   TENSORRT_INCLUDE_DIR    (or TENSORRT_INCLUDE) TensorRT headers, forwarded to CMake.
//!   FB_TRT_LEAN_DLL         nvinfer_lean_11.dll, copied next to the exe (dev convenience).
//!   FFMPEG_DIR              FFmpeg SDK root; FFMPEG_DIR/bin/*.dll copied next to the exe.
//!   CUDA_PATH               cudart64_*.dll copied next to the exe when present.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let rd = fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else if p.file_name().map(|n| n.eq_ignore_ascii_case(name)).unwrap_or(false) {
            return Some(p);
        }
    }
    for d in subdirs {
        if let Some(f) = find_file(&d, name) {
            return Some(f);
        }
    }
    None
}

fn copy_to_dirs(src: &Path, dirs: &[PathBuf]) {
    let Some(name) = src.file_name() else { return };
    for d in dirs {
        let _ = fs::create_dir_all(d);
        let dst = d.join(name);
        // Skip if identical size+mtime to avoid needless churn.
        if let (Ok(a), Ok(b)) = (fs::metadata(src), fs::metadata(&dst)) {
            if a.len() == b.len() && b.modified().ok() >= a.modified().ok() {
                continue;
            }
        }
        if let Err(e) = fs::copy(src, &dst) {
            println!("cargo:warning=could not copy {} -> {}: {}", src.display(), dst.display(), e);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let native = manifest.join("native");
    let header = native.join("include").join("fb_native.h");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native");
    for v in [
        "FB_SKIP_NATIVE",
        "TENSORRT_INCLUDE_DIR",
        "TENSORRT_INCLUDE",
        "FB_TRT_LEAN_DLL",
        "FFMPEG_DIR",
        "CUDA_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    // ---------------- bindgen ----------------
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .allowlist_function("fb_.*")
        .allowlist_type("fb_.*")
        .allowlist_var("FB_.*")
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed on fb_native.h (is LIBCLANG_PATH set?)");
    bindings
        .write_to_file(out_dir.join("fb_native_bindings.rs"))
        .expect("write bindings");

    if env::var("FB_SKIP_NATIVE").map(|v| v == "1").unwrap_or(false) {
        println!("cargo:warning=FB_SKIP_NATIVE=1: fb_native is NOT built or linked");
        return;
    }

    // ---------------- CMake ----------------
    let mut cfg = cmake::Config::new(&native);
    cfg.generator("Ninja").profile("Release").build_target("fb_native");
    if let Ok(trt) = env::var("TENSORRT_INCLUDE_DIR").or_else(|_| env::var("TENSORRT_INCLUDE")) {
        cfg.define("TENSORRT_INCLUDE_DIR", trt);
    }
    if let Ok(cuda) = env::var("CUDA_PATH") {
        cfg.define("CUDAToolkit_ROOT", cuda.replace('\\', "/"));
    }
    // Make Ninja/MSVC pick Release flags regardless of the cargo profile.
    cfg.define("CMAKE_BUILD_TYPE", "Release");
    let dst = cfg.build();

    let lib = find_file(&dst, "fb_native.lib").or_else(|| find_file(&out_dir, "fb_native.lib"));
    let dll = find_file(&dst, "fb_native.dll").or_else(|| find_file(&out_dir, "fb_native.dll"));
    let (lib, dll) = match (lib, dll) {
        (Some(l), Some(d)) => (l, d),
        _ => panic!(
            "CMake build finished but fb_native.lib / fb_native.dll not found under {}",
            dst.display()
        ),
    };
    println!("cargo:rustc-link-search=native={}", lib.parent().unwrap().display());
    println!("cargo:rustc-link-lib=dylib=fb_native");

    // ---------------- copy runtime DLLs ----------------
    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out
    let profile_dir = out_dir.ancestors().nth(3).expect("profile dir").to_path_buf();
    let dirs = vec![profile_dir.clone(), profile_dir.join("deps"), profile_dir.join("examples")];

    copy_to_dirs(&dll, &dirs);
    if let Ok(trt) = env::var("FB_TRT_LEAN_DLL") {
        let p = PathBuf::from(trt);
        if p.is_file() {
            copy_to_dirs(&p, &dirs);
        }
    }
    if let Ok(ff) = env::var("FFMPEG_DIR") {
        let bin = Path::new(&ff).join("bin");
        if let Ok(rd) = fs::read_dir(&bin) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().map(|x| x.eq_ignore_ascii_case("dll")).unwrap_or(false) {
                    copy_to_dirs(&p, &dirs);
                }
            }
        }
    }
    if let Ok(cuda) = env::var("CUDA_PATH") {
        if let Ok(rd) = fs::read_dir(Path::new(&cuda).join("bin")) {
            for e in rd.flatten() {
                let p = e.path();
                let n = p.file_name().unwrap().to_string_lossy().to_lowercase();
                if n.starts_with("cudart64_") && n.ends_with(".dll") {
                    copy_to_dirs(&p, &dirs);
                }
            }
        }
    }
}
