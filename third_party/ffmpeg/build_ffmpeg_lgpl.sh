#!/usr/bin/env bash
# =============================================================================
# build_ffmpeg_lgpl.sh - minimal, LGPL-2.1+, *shared* FFmpeg build for filming-better
#
# What it produces (default: third_party/ffmpeg/dist/):
#   bin/avcodec-62.dll avformat-62.dll avutil-60.dll swscale-9.dll swresample-6.dll
#   lib/*.lib (MSVC import libs)  include/  BUILD_CONFIG.txt (configure line, for LGPL compliance)
#
# Licence guard ("Dinh GPL" risk in docs/blueprint.md): the script FAILS if the resulting
# configuration contains --enable-gpl / --enable-version3 / --enable-nonfree / x264 / x265 /
# any other GPL-only lib, both on config.mak/config.h (after configure) and on the built
# avutil_license() string (after make install).
#
# Toolchain: MSYS2 + MSVC (cl.exe). Run from an "MSYS2 MSYS" shell that has the MSVC
# environment, the easy way is:   third_party\ffmpeg\build_ffmpeg_lgpl.cmd
# (it calls vcvars64.bat, then bash from MSYS2). Prerequisites inside MSYS2:
#     pacman -S --needed make diffutils git pkgconf nasm
# and MSYS2's /usr/bin/link.exe must NOT shadow MSVC link.exe:
#     mv /usr/bin/link.exe /usr/bin/link.exe.bak
#
# Usage:
#   build_ffmpeg_lgpl.sh [--dry-run] [--jobs N] [--work DIR] [--prefix DIR] [--verify-only]
#     --dry-run      print the configure command (+ what would be cloned/built), run nothing
#     --verify-only  only run the licence checks on an existing build in --work/--prefix
# Environment overrides:
#   FFMPEG_TAG (n8.0.1)  NVCODEC_TAG (n13.0.19.1)  FFMPEG_SRC (use an existing source tree)
#   FB_FFMPEG_DAV1D=1    also link external libdav1d (BSD-2, LGPL-compatible) for SW AV1 decode;
#                        needs a dav1d.pc on PKG_CONFIG_PATH. Default OFF: AV1 decode = NVDEC only
#                        (RTX 30+; Turing/RTX 20 has no AV1 decoder).
#   FB_FFMPEG_AVFILTER=1 build libavfilter too with a tiny filter set (null,anull,format,aformat,
#                        scale,fps,hwupload,hwdownload). Default OFF.
#
# Version pins (keep in sync with docs/contracts.md and app/):
#   FFmpeg n8.0.1   -> avcodec-62 / avformat-62 / avutil-60 / swscale-9 / swresample-6
#                      (same ABI as the dev build in C:/ffmpeg-8.0.1-full_build-shared)
#   nv-codec-headers n13.0.19.1 (Video Codec SDK 13.0.19) -> requires NVIDIA driver >= 570.0
#                      on Windows/Linux (NVENC API 13.0). This equals the CUDA 12.8 driver floor
#                      (FB_MIN_DRIVER_VERSION 12080 in app/native/include/fb_native.h), so the
#                      app-wide minimum driver is R570. (n12.2.72.0 would be 551.76; not used because
#                      cu12.8 already forces R570 and SDK 13 is the one FFmpeg 8 targets.)
# =============================================================================
set -euo pipefail

FFMPEG_TAG="${FFMPEG_TAG:-n8.0.1}"
FFMPEG_GIT="${FFMPEG_GIT:-https://github.com/FFmpeg/FFmpeg.git}"
NVCODEC_TAG="${NVCODEC_TAG:-n13.0.19.1}"
NVCODEC_GIT="${NVCODEC_GIT:-https://github.com/FFmpeg/nv-codec-headers.git}"
NVCODEC_MIN_DRIVER_WIN="570.0"
EXPECTED_DLLS=(avcodec-62 avformat-62 avutil-60 swscale-9 swresample-6)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HERE/build"
PREFIX="$HERE/dist"
JOBS="$(nproc 2>/dev/null || echo 4)"
DRY=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --jobs) JOBS="$2"; shift ;;
    --work) WORK="$2"; shift ;;
    --prefix) PREFIX="$2"; shift ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

log()  { printf '\n==> %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
join() { local IFS=,; echo "$*"; }

# ---------------------------------------------------------------------------
# Component lists (all names validated against the n8.0.1 source: allformats.c,
# allcodecs.c, bitstream_filters.c, parsers.c, protocols.c; re-validated at build time)
# ---------------------------------------------------------------------------
PROTOCOLS=(file pipe)
# mov = mp4/mov/m4a/3gp ; matroska = mkv/webm ; h264/hevc/ivf/obu = raw elementary streams
DEMUXERS=(mov matroska mpegts avi h264 hevc ivf obu)
# outputs per blueprint: H.264/HEVC/AV1 in mp4/mov/mkv ; mpegts + image2 (PNG preview) are cheap extras
MUXERS=(mp4 mov matroska webm mpegts image2 null)
# Stream-copy of audio/video needs parsers to find frame boundaries / timestamps.
PARSERS=(h264 hevc av1 vp9 vp8 aac aac_latm ac3 dca flac mpegaudio opus vorbis mpeg4video mpegvideo)
BSFS=(aac_adtstoasc extract_extradata h264_mp4toannexb hevc_mp4toannexb
      av1_frame_merge av1_frame_split av1_metadata h264_metadata hevc_metadata vp9_metadata
      vp9_superframe vp9_superframe_split vp9_raw_reorder dts2pts setts null
      mov2textsub text2movsub opus_metadata pgs_frame_merge)
# Native decoders are LGPL. AV1 native decoder = hwaccel only (nvdec); *_cuvid are the NVDEC
# stand-alone decoders (frames stay in VRAM). Subtitle decoders/encoders are only used to convert
# text subs (srt/ass/webvtt) to mov_text when muxing to MP4/MOV. png/rawvideo: preview + testing.
DECODERS=(h264 hevc av1 vp9 h264_cuvid hevc_cuvid av1_cuvid vp9_cuvid
          subrip srt ass ssa movtext webvtt png rawvideo)
ENCODERS=(h264_nvenc hevc_nvenc av1_nvenc movtext subrip srt ass ssa webvtt png rawvideo)
HWACCELS=(h264_nvdec hevc_nvdec av1_nvdec vp9_nvdec)

CONFIGURE_ARGS=(
  --toolchain=msvc --arch=x86_64 --target-os=win64
  "--prefix=$PREFIX"
  # --- licence: LGPL v2.1+ only. (--enable-gpl / --enable-version3 / --enable-nonfree are NEVER passed)
  --disable-gpl --disable-nonfree --disable-version3
  # --- shared only, no CLI tools
  --enable-shared --disable-static
  --disable-programs --disable-doc --disable-debug
  # --- libraries: avcodec avformat avutil swscale swresample only (libpostproc no longer exists in FFmpeg 8)
  --disable-avdevice --disable-network
  # --- nothing picked up from the host (zlib, bzlib, lzma, iconv, schannel, mediafoundation, d3d11va...)
  --disable-autodetect
  --disable-everything
  "--enable-protocol=$(join "${PROTOCOLS[@]}")"
  "--enable-demuxer=$(join "${DEMUXERS[@]}")"
  "--enable-muxer=$(join "${MUXERS[@]}")"
  "--enable-parser=$(join "${PARSERS[@]}")"
  "--enable-bsf=$(join "${BSFS[@]}")"
  "--enable-decoder=$(join "${DECODERS[@]}")"
  "--enable-encoder=$(join "${ENCODERS[@]}")"
  "--enable-hwaccel=$(join "${HWACCELS[@]}")"
  # --- NVIDIA: dynamic-loaded (no CUDA SDK / nvcc needed; --enable-cuda-nvcc and --enable-libnpp are
  #     "nonfree" in FFmpeg and are deliberately NOT used)
  --enable-ffnvcodec --enable-cuda --enable-cuvid --enable-nvdec --enable-nvenc
  --disable-cuda-llvm
)
if [[ "${FB_FFMPEG_AVFILTER:-0}" == "1" ]]; then
  CONFIGURE_ARGS+=(--enable-avfilter
    "--enable-filter=null,anull,format,aformat,scale,fps,hwupload,hwdownload")
else
  CONFIGURE_ARGS+=(--disable-avfilter)
fi
if [[ "${FB_FFMPEG_DAV1D:-0}" == "1" ]]; then
  CONFIGURE_ARGS+=(--enable-libdav1d --enable-decoder=libdav1d)
fi

print_configure() {
  printf './configure'
  local a
  for a in "${CONFIGURE_ARGS[@]}"; do
    if [[ "$a" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then printf ' \\\n    %s' "$a"; else printf ' \\\n    %q' "$a"; fi
  done
  printf '\n'
}

# ---------------------------------------------------------------------------
# Licence checks
# ---------------------------------------------------------------------------
BAD_RE='--enable-(gpl|version3|nonfree|libx264|libx265|libxvid|libfdk[-_]aac|libvidstab|libzimg|libxavs2?|libdavs2|librubberband|libcdio|frei0r|avisynth|postproc|libaribb24|libsmbclient|decklink|openssl|cuda-nvcc|libnpp)'
BAD_CFG='CONFIG_(GPL|GPLV3|VERSION3|NONFREE|LIBX264|LIBX265|LIBXVID|LIBFDK_AAC|LIBVIDSTAB|LIBZIMG|LIBXAVS2?|LIBDAVS2|FREI0R|AVISYNTH|POSTPROC|LIBNPP|CUDA_NVCC)'

check_config_license() {   # $1 = ffmpeg build dir (after configure)
  local d="$1" cfg="$1/ffbuild/config.mak" hdr="$1/config.h" bad=0
  [[ -f "$cfg" && -f "$hdr" ]] || die "configure output not found in $d"
  # config.mak: enabled options are 'CONFIG_X=yes', disabled ones are '!CONFIG_X=yes'
  if grep -Eq "^${BAD_CFG}=yes" "$cfg"; then
    echo "  offending lines in ffbuild/config.mak:"; grep -E "^${BAD_CFG}=yes" "$cfg" | sed 's/^/    /'
    bad=1
  fi
  if grep -Eq '^#define FFMPEG_CONFIGURATION .*--enable-(gpl|version3|nonfree|libx26[45])' "$hdr"; then
    echo "  FFMPEG_CONFIGURATION in config.h contains a GPL/nonfree flag"; bad=1
  fi
  local lic; lic="$(sed -n 's/^#define FFMPEG_LICENSE "\(.*\)"/\1/p' "$hdr")"
  echo "  FFMPEG_LICENSE (config.h): '$lic'"
  [[ "$lic" == "LGPL version 2.1 or later" ]] || { echo "  expected 'LGPL version 2.1 or later'"; bad=1; }
  # required hardware components must really be on
  local must=(CONFIG_H264_NVENC_ENCODER CONFIG_HEVC_NVENC_ENCODER CONFIG_AV1_NVENC_ENCODER
              CONFIG_H264_NVDEC_HWACCEL CONFIG_HEVC_NVDEC_HWACCEL CONFIG_AV1_NVDEC_HWACCEL
              CONFIG_VP9_NVDEC_HWACCEL CONFIG_CUVID CONFIG_NVDEC CONFIG_NVENC CONFIG_FFNVCODEC
              CONFIG_MOV_DEMUXER CONFIG_MATROSKA_DEMUXER CONFIG_MPEGTS_DEMUXER
              CONFIG_MP4_MUXER CONFIG_MATROSKA_MUXER CONFIG_MOV_MUXER)
  local m
  for m in "${must[@]}"; do
    # enabled components are 'CONFIG_X=yes' in ffbuild/config.mak (config_components.h holds the 0/1 defines)
    grep -Eq "^$m=yes" "$cfg" || { echo "  required component not enabled: $m"; bad=1; }
  done
  [[ $bad -eq 0 ]] || die "LICENCE/COMPONENT CHECK FAILED (configure stage) - refusing to build"
  echo "  OK: LGPL-only configuration, NVENC/NVDEC components present"
}

check_built_license() {    # $1 = prefix ; compiles a tiny probe against avutil.lib
  local pfx="$1" probe="$WORK/license_probe"
  mkdir -p "$probe"
  cat > "$probe/probe.c" <<'CEOF'
#include <stdio.h>
#include <string.h>
#include <libavutil/avutil.h>
#include <libavcodec/avcodec.h>
int main(void) {
    const char *lic = avutil_license();
    const char *cfg = avcodec_configuration();
    printf("LICENSE: %s\nCONFIGURATION: %s\n", lic, cfg);
    if (strncmp(lic, "LGPL version 2.1", 16) != 0) return 10;
    if (strstr(cfg, "--enable-gpl") || strstr(cfg, "--enable-nonfree") ||
        strstr(cfg, "--enable-version3") || strstr(cfg, "x264") || strstr(cfg, "x265")) return 11;
    return 0;
}
CEOF
  command -v cl >/dev/null 2>&1 || die "cl.exe not on PATH (needed for the built-DLL licence probe)"
  ( cd "$probe" && export MSYS2_ARG_CONV_EXCL='*' && PATH="$pfx/bin:$PATH" && \
    cl -nologo -I"$(cygpath -m "$pfx/include")" probe.c \
       -link -LIBPATH:"$(cygpath -m "$pfx/lib")" avutil.lib avcodec.lib -OUT:probe.exe >/dev/null ) \
    || die "could not compile licence probe"
  local out rc=0
  out="$(cd "$probe" && PATH="$pfx/bin:$PATH" ./probe.exe)" || rc=$?
  echo "$out" | sed 's/^/  /'
  [[ $rc -eq 0 ]] || die "BUILT LIBRARIES ARE NOT LGPL-CLEAN (probe exit code $rc)"
  # belt and braces: scan the DLLs themselves for GPL-only library markers
  local f
  for f in "$pfx"/bin/*.dll; do
    if grep -aqE 'x264 - core|x265 HEVC|--enable-gpl' "$f"; then die "GPL marker found in $(basename "$f")"; fi
  done
  echo "  OK: built DLLs report LGPL, no x264/x265 markers"
}

# ---------------------------------------------------------------------------
if [[ $DRY -eq 1 ]]; then
  cat <<EOT
[dry-run] FFmpeg $FFMPEG_TAG from $FFMPEG_GIT
[dry-run] nv-codec-headers $NVCODEC_TAG from $NVCODEC_GIT  (needs NVIDIA driver >= $NVCODEC_MIN_DRIVER_WIN)
[dry-run] work dir : $WORK
[dry-run] prefix   : $PREFIX
[dry-run] jobs     : $JOBS
[dry-run] steps    : clone nv-codec-headers -> make install PREFIX=$WORK/deps
                     clone FFmpeg -> validate component names -> configure -> licence check
                     make -j$JOBS -> make install -> licence probe on DLLs -> BUILD_CONFIG.txt
[dry-run] PKG_CONFIG_PATH=$WORK/deps/lib/pkgconfig
[dry-run] configure command (run inside the FFmpeg source dir, from an MSVC-enabled MSYS2 shell):
EOT
  print_configure
  echo
  echo "[dry-run] expected DLLs: ${EXPECTED_DLLS[*]/%/.dll}"
  # Static self-check of the pass-list: nothing GPL may be in the argument list.
  if printf '%s\n' "${CONFIGURE_ARGS[@]}" | grep -Eq -- "$BAD_RE"; then
    die "configure argument list contains a GPL/nonfree option"
  fi
  echo "[dry-run] OK: argument list is free of --enable-gpl/-version3/-nonfree/x264/x265"
  exit 0
fi

# ---------------------------------------------------------------------------
# Real build
# ---------------------------------------------------------------------------
if printf '%s\n' "${CONFIGURE_ARGS[@]}" | grep -Eq -- "$BAD_RE"; then
  die "configure argument list contains a GPL/nonfree option"
fi

if [[ $VERIFY_ONLY -eq 1 ]]; then
  log "Verify-only: licence checks on $WORK / $PREFIX"
  check_config_license "$WORK/obj"
  check_built_license "$PREFIX"
  exit 0
fi

log "Checking tools"
for t in git make nasm pkg-config cl; do
  command -v "$t" >/dev/null 2>&1 || die "'$t' not found. In MSYS2: pacman -S make diffutils git pkgconf nasm; run from a shell with vcvars64 (cl.exe) on PATH"
done
lnk="$(command -v link || true)"
case "$lnk" in
  /usr/bin/link*) die "MSYS2's /usr/bin/link.exe shadows MSVC link.exe. Fix: mv /usr/bin/link.exe /usr/bin/link.exe.bak" ;;
esac

mkdir -p "$WORK" "$PREFIX"
DEPS="$WORK/deps"

log "nv-codec-headers $NVCODEC_TAG (driver >= $NVCODEC_MIN_DRIVER_WIN)"
if [[ ! -d "$WORK/nv-codec-headers/.git" ]]; then
  git clone --depth 1 --branch "$NVCODEC_TAG" "$NVCODEC_GIT" "$WORK/nv-codec-headers"
fi
( cd "$WORK/nv-codec-headers" && make install PREFIX="$DEPS" )
export PKG_CONFIG_PATH="$DEPS/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
pkg-config --exists ffnvcodec || die "ffnvcodec.pc not found in $DEPS/lib/pkgconfig"
echo "  ffnvcodec $(pkg-config --modversion ffnvcodec)"

log "FFmpeg source $FFMPEG_TAG"
SRC="${FFMPEG_SRC:-$WORK/ffmpeg}"
if [[ ! -d "$SRC/.git" && ! -f "$SRC/configure" ]]; then
  git clone --depth 1 --branch "$FFMPEG_TAG" "$FFMPEG_GIT" "$SRC"
fi
FFMPEG_COMMIT="$(cd "$SRC" && git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  commit $FFMPEG_COMMIT"

log "Validating component names against the source tree"
validate() { # file kind-suffix names...
  local f="$SRC/$1" suf="$2"; shift 2; local n miss=0
  [[ -f "$f" ]] || die "missing $f"
  for n in "$@"; do grep -Eq "ff_${n}_${suf}\b" "$f" || { echo "  unknown $suf: $n"; miss=1; }; done
  return $miss
}
ok=0
validate libavformat/allformats.c demuxer  "${DEMUXERS[@]}"   || ok=1
validate libavformat/allformats.c muxer    "${MUXERS[@]}"     || ok=1
validate libavformat/protocols.c  protocol "${PROTOCOLS[@]}"  || ok=1
validate libavcodec/parsers.c     parser   "${PARSERS[@]}"    || ok=1
validate libavcodec/bitstream_filters.c bsf "${BSFS[@]}"      || ok=1
validate libavcodec/allcodecs.c   decoder  "${DECODERS[@]}"   || ok=1
validate libavcodec/allcodecs.c   encoder  "${ENCODERS[@]}"   || ok=1
[[ $ok -eq 0 ]] || die "component list does not match FFmpeg $FFMPEG_TAG (see above)"

log "configure"
mkdir -p "$WORK/obj" && cd "$WORK/obj"
# out-of-tree build: ./configure lives in the source dir, config.* land in $WORK/obj
print_configure | sed "1s#^\./configure#$SRC/configure#" > "$WORK/configure_cmd.txt"
"$SRC/configure" "${CONFIGURE_ARGS[@]}"

log "Licence check on configuration"
check_config_license "$WORK/obj"

log "make -j$JOBS"
make -j"$JOBS"
make install

log "Checking produced DLLs"
for d in "${EXPECTED_DLLS[@]}"; do
  [[ -f "$PREFIX/bin/$d.dll" ]] || die "missing $PREFIX/bin/$d.dll"
  echo "  $d.dll  $(stat -c %s "$PREFIX/bin/$d.dll") bytes"
done

log "Licence check on built libraries"
check_built_license "$PREFIX"

log "Writing BUILD_CONFIG.txt"
LICENSE_LINE="$(sed -n 's/^#define FFMPEG_LICENSE "\(.*\)"/\1/p' "$WORK/obj/config.h")"
CL_VER="$(cl 2>&1 | head -1 | tr -d '\r')"
cat > "$PREFIX/BUILD_CONFIG.txt" <<EOT
FFmpeg build configuration (LGPL compliance information) - filming-better
===========================================================================
Libraries : avcodec avformat avutil swscale swresample  (dynamically linked, shared DLLs)
License   : $LICENSE_LINE   (GNU LGPL v2.1 or later; --enable-gpl / --enable-version3 / --enable-nonfree NOT used)
Source    : $FFMPEG_GIT  tag $FFMPEG_TAG  commit $FFMPEG_COMMIT
            (unmodified upstream; tarball: https://ffmpeg.org/releases/ffmpeg-${FFMPEG_TAG#n}.tar.xz)
nv-codec-headers (MIT, headers only; the NVIDIA driver provides nvcuvid/nvEncodeAPI at runtime):
            $NVCODEC_GIT  tag $NVCODEC_TAG   (minimum NVIDIA driver $NVCODEC_MIN_DRIVER_WIN)
Compiler  : $CL_VER
Built on  : $(date -u +%Y-%m-%dT%H:%M:%SZ)
Script    : third_party/ffmpeg/build_ffmpeg_lgpl.sh in the filming-better repository

Configure command
-----------------
$(cat "$WORK/configure_cmd.txt")

Resulting avcodec_configuration()
---------------------------------
$(cd "$WORK/license_probe" && PATH="$PREFIX/bin:$PATH" ./probe.exe | sed -n 's/^CONFIGURATION: //p')

Relinking (LGPL section 6): you may replace the FFmpeg DLLs next to filming-better.exe
with your own build of the same FFmpeg major versions (avcodec-62 avformat-62 avutil-60
swscale-9 swresample-6) created with the command above (or any compatible configuration).
EOT
cp "$PREFIX/BUILD_CONFIG.txt" "$HERE/BUILD_CONFIG.txt"

log "DONE. Install tree: $PREFIX"
echo "  Point the Rust build at it:  FFMPEG_DIR=$(cygpath -m "$PREFIX" 2>/dev/null || echo "$PREFIX")"
