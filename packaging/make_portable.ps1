<#
.SYNOPSIS
  Assembles the portable filming-better package and zips it:
    dist\filming-better-<version>-win64\   +   dist\filming-better-<version>-win64.zip

.DESCRIPTION
  Package content:
    filming-better.exe, fb_native.dll, nvinfer_lean_11.dll (TensorRT LEAN runtime only - the
    builder DLLs nvinfer_11.dll / nvinfer_builder_resource_*.dll are never copied),
    LGPL FFmpeg DLLs (avcodec/avformat/avutil/swscale/swresample + their non-system dependencies),
    app-local MSVC runtime DLLs if the binaries import them,
    models\manifest.json + the default engine set (span_x2 + rife_v4) as the PORTABLE `ampere+` engines
    (they run on every GPU with cc >= 8.0; the app prefers a per-GPU `sm<cc>` engine when one is installed,
    e.g. fetched with `models pull`, and falls back to `ampere+` otherwise, so the manifest may list sm86 engines
    that are not in the package),
    LICENSE, THIRD_PARTY_NOTICES, README.md, BUILD_CONFIG.txt (FFmpeg configure line, LGPL).

  Checks (exit code != 0 when violated):
    * required files exist; engine sha256 matches models\manifest.json
    * shipped FFmpeg DLLs are LGPL (no --enable-gpl / x264 / x265 markers) unless -AllowDevFFmpeg
    * forbidden TensorRT builder DLLs are not in the package
    * zip size <= -MaxZipMB (default 200, blueprint "ngan sach dung luong")
    * optional installer size <= -MaxInstallerMB (default 70, blueprint 4.4). NOTE: the blueprint's
      70 MB (installer) and 200 MB (portable zip incl. engines) limits only make sense together if
      the installer does NOT embed the engines (they are fetched with `filming-better models pull`).
      installer.iss therefore ships without engines by default.

  Works on Windows PowerShell 5.1 and PowerShell 7.

.PARAMETER Arches
  Engine arch tags to bundle. Default `ampere+` (portable, one engine for every Ampere+ GPU, ~15-50% slower
  than a per-GPU engine). Per-GPU engines are NOT bundled by default; add them explicitly, e.g.
  `-Arches ampere+,sm86` (keep ampere+ as the fallback). Valid: ampere+ and sm<major><minor> (sm75, sm86 ...).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File packaging\make_portable.ps1
.EXAMPLE
  # CI: version from tag, installer check
  packaging\make_portable.ps1 -Version 1.0.0 -RepoOwner myorg -InstallerPath dist\filming-better-1.0.0-setup.exe
.EXAMPLE
  # local smoke test with the GPL dev FFmpeg (never publish the result)
  packaging\make_portable.ps1 -AllowDevFFmpeg -MaxZipMB 1000
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$ReleaseDir,                 # default: <repo>\app\target\release
    [string]$FFmpegDir,                  # default: <repo>\third_party\ffmpeg\dist\bin
    [string]$TrtLeanDll,                 # default: $env:FB_TRT_LEAN_DLL, ReleaseDir, .venv tensorrt_lean_libs
    [string]$EnginesDir,                 # default: <repo>\models\engines
    [string]$ManifestPath,               # default: <repo>\models\manifest.json
    [string[]]$Engines = @('span_x2', 'rife_v4'),
    [string[]]$Arches  = @('ampere+'),
    [string]$BuildConfig,                # FFmpeg BUILD_CONFIG.txt
    [string]$OutDir,                     # default: <repo>\dist
    [string]$RepoOwner,                  # replaces OWNER in the shipped manifest base_url
    [int]$MaxZipMB = 200,
    [int]$MaxInstallerMB = 70,
    [string]$InstallerPath,              # if the file exists its size is checked against -MaxInstallerMB
    [switch]$AllowDevFFmpeg,             # allow GPL dev FFmpeg DLLs (local testing ONLY)
    [switch]$SkipZip,
    [switch]$CheckSizesOnly,             # only check the size of an existing zip (+ installer) and exit
    [switch]$Strict                      # placeholder hashes in the manifest become errors
)

Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $ReleaseDir)   { $ReleaseDir   = Join-Path $Repo 'app\target\release' }
if (-not $EnginesDir)   { $EnginesDir   = Join-Path $Repo 'models\engines' }
if (-not $ManifestPath) { $ManifestPath = Join-Path $Repo 'models\manifest.json' }
if (-not $OutDir)       { $OutDir       = Join-Path $Repo 'dist' }
$MB = 1048576.0
foreach ($a in $Arches) {
    if ($a -ne 'ampere+' -and $a -notmatch '^sm\d{2,3}$') { Write-Host ("ERROR: bad -Arches entry '" + $a + "' (use ampere+ or sm<major><minor>)") -ForegroundColor Red; exit 2 }
}

function Fail([string]$msg, [int]$code = 1) {
    Write-Host ("ERROR: " + $msg) -ForegroundColor Red
    exit $code
}
function Info([string]$msg) { Write-Host $msg }
function Warn([string]$msg) { Write-Host ("WARNING: " + $msg) -ForegroundColor Yellow }
function Section([string]$msg) { Write-Host ""; Write-Host ("==> " + $msg) -ForegroundColor Cyan }
function FmtMB([double]$bytes) { return ('{0,8:N2} MB' -f ($bytes / $MB)) }

# ---------------------------------------------------------------------------------------------
# Small C# helpers: PE import table reader + streaming ASCII search (PowerShell 5 = C# 5 syntax)
# ---------------------------------------------------------------------------------------------
if (-not ('FbPe' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;

public static class FbPe
{
    private static long RvaToOff(uint rva, List<uint[]> secs)
    {
        foreach (uint[] s in secs)
        {
            uint va = s[0], vsize = s[1], rawptr = s[2], rawsize = s[3];
            uint span = Math.Max(vsize, rawsize);
            if (rva >= va && rva < va + span) return (long)rawptr + (rva - va);
        }
        return -1;
    }

    // Names of the DLLs in the (normal) import table of a PE file.
    public static string[] Imports(string path)
    {
        List<string> res = new List<string>();
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (BinaryReader br = new BinaryReader(fs))
        {
            fs.Seek(0x3C, SeekOrigin.Begin);
            int pe = br.ReadInt32();
            fs.Seek(pe, SeekOrigin.Begin);
            if (br.ReadUInt32() != 0x00004550) throw new InvalidDataException("not a PE file: " + path);
            br.ReadUInt16();                    // machine
            int nsec = br.ReadUInt16();
            fs.Seek(12, SeekOrigin.Current);    // timestamp, symbol table, number of symbols
            int optSize = br.ReadUInt16();
            br.ReadUInt16();                    // characteristics
            long optStart = fs.Position;
            ushort magic = br.ReadUInt16();
            int ddOff = (magic == 0x20B) ? 112 : 96;
            fs.Seek(optStart + ddOff + 8, SeekOrigin.Begin);   // data directory[1] = imports
            uint impRva = br.ReadUInt32();
            long secStart = optStart + optSize;
            List<uint[]> secs = new List<uint[]>();
            for (int i = 0; i < nsec; i++)
            {
                fs.Seek(secStart + i * 40 + 8, SeekOrigin.Begin);
                uint vsize = br.ReadUInt32();
                uint va = br.ReadUInt32();
                uint rawsize = br.ReadUInt32();
                uint rawptr = br.ReadUInt32();
                secs.Add(new uint[] { va, vsize, rawptr, rawsize });
            }
            if (impRva == 0) return res.ToArray();
            long off = RvaToOff(impRva, secs);
            if (off < 0) return res.ToArray();
            for (int n = 0; n < 4096; n++)
            {
                fs.Seek(off + n * 20, SeekOrigin.Begin);
                br.ReadUInt32(); br.ReadUInt32(); br.ReadUInt32();
                uint nameRva = br.ReadUInt32();
                br.ReadUInt32();
                if (nameRva == 0) break;
                long nameOff = RvaToOff(nameRva, secs);
                if (nameOff < 0) break;
                fs.Seek(nameOff, SeekOrigin.Begin);
                StringBuilder sb = new StringBuilder();
                int c;
                while ((c = fs.ReadByte()) > 0 && sb.Length < 260) sb.Append((char)c);
                res.Add(sb.ToString());
            }
        }
        return res.ToArray();
    }

    public static bool ContainsAscii(string path, string needle)
    {
        byte[] pat = Encoding.ASCII.GetBytes(needle);
        byte[] buf = new byte[4 * 1024 * 1024 + pat.Length];
        using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            int carry = 0;
            while (true)
            {
                int n = fs.Read(buf, carry, buf.Length - carry);
                if (n <= 0) break;
                int total = carry + n;
                int last = total - pat.Length;
                for (int i = 0; i <= last; i++)
                {
                    if (buf[i] != pat[0]) continue;
                    int j = 1;
                    while (j < pat.Length && buf[i + j] == pat[j]) j++;
                    if (j == pat.Length) return true;
                }
                carry = Math.Min(pat.Length - 1, total);
                Array.Copy(buf, total - carry, buf, 0, carry);
            }
        }
        return false;
    }

    // Returns the text after "needle" up to the next NUL (used to read FFmpeg's configuration string)
    public static string AsciiAfter(string path, string needle, int max)
    {
        byte[] data = File.ReadAllBytes(path);
        byte[] pat = Encoding.ASCII.GetBytes(needle);
        for (int i = 0; i + pat.Length < data.Length; i++)
        {
            if (data[i] != pat[0]) continue;
            int j = 1;
            while (j < pat.Length && data[i + j] == pat[j]) j++;
            if (j != pat.Length) continue;
            StringBuilder sb = new StringBuilder();
            for (int k = i; k < data.Length && data[k] != 0 && sb.Length < max; k++) sb.Append((char)data[k]);
            return sb.ToString();
        }
        return null;
    }
}
'@
}

# ---------------------------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------------------------
if (-not $Version) {
    $cargo = Get-Content (Join-Path $Repo 'app\Cargo.toml') -ErrorAction Stop
    foreach ($l in $cargo) {
        if ($l -match '^\s*version\s*=\s*"([^"]+)"') { $Version = $Matches[1]; break }
    }
    if (-not $Version) { Fail 'could not determine version (pass -Version)' }
}
$Version = $Version.TrimStart('v')
$PkgName = 'filming-better-' + $Version + '-win64'
$PkgDir  = Join-Path $OutDir $PkgName
$ZipPath = Join-Path $OutDir ($PkgName + '.zip')

Section ("filming-better portable package " + $Version)
Info ("repo        : " + $Repo)
Info ("release dir : " + $ReleaseDir)

function Test-Sizes {
    # returns 0 = ok, 3 = zip too big, 4 = installer too big
    $rc = 0
    if (Test-Path $ZipPath) {
        $zb = (Get-Item $ZipPath).Length
        Info ("  ZIP       : " + (FmtMB $zb) + "   limit " + $MaxZipMB + " MB   (" + $ZipPath + ")")
        if ($zb -gt ($MaxZipMB * $MB)) {
            Write-Host ("ERROR: portable zip is " + [math]::Round($zb / $MB, 2) + " MB > " + $MaxZipMB + " MB limit") -ForegroundColor Red
            $rc = 3
        } else { Info '  OK: zip is within the size budget' }
    } elseif ($CheckSizesOnly) {
        Fail ("zip not found: " + $ZipPath)
    }
    if ($InstallerPath) {
        if (Test-Path $InstallerPath) {
            $ib = (Get-Item $InstallerPath).Length
            Info ("  Installer : " + (FmtMB $ib) + "   limit " + $MaxInstallerMB + " MB   (" + $InstallerPath + ")")
            if ($ib -gt ($MaxInstallerMB * $MB)) {
                Write-Host ("ERROR: installer is " + [math]::Round($ib / $MB, 2) + " MB > " + $MaxInstallerMB + " MB limit") -ForegroundColor Red
                if ($rc -eq 0) { $rc = 4 }
            } else { Info '  OK: installer is within the size budget' }
        } elseif ($CheckSizesOnly) {
            Fail ("installer not found: " + $InstallerPath)
        } else {
            Warn ("installer not found yet, size check skipped: " + $InstallerPath)
        }
    }
    return $rc
}

if ($CheckSizesOnly) {
    Section 'Size check only'
    exit (Test-Sizes)
}

# ---------------------------------------------------------------------------------------------
# Locate inputs
# ---------------------------------------------------------------------------------------------
$exe = Join-Path $ReleaseDir 'filming-better.exe'
$fbn = Join-Path $ReleaseDir 'fb_native.dll'
if (-not (Test-Path $exe)) { Fail ("missing " + $exe + "  (build: cargo build --release --manifest-path app/Cargo.toml)") }
if (-not (Test-Path $fbn)) { Fail ("missing " + $fbn) }

# TensorRT lean runtime ------------------------------------------------------------------------
$leanCandidates = @()
if ($TrtLeanDll) { $leanCandidates += $TrtLeanDll }
if ($env:FB_TRT_LEAN_DLL) { $leanCandidates += $env:FB_TRT_LEAN_DLL }
$leanCandidates += (Join-Path $ReleaseDir 'nvinfer_lean_11.dll')
$leanCandidates += (Join-Path $Repo '.venv\Lib\site-packages\tensorrt_lean_libs\nvinfer_lean_11.dll')
$lean = $null
foreach ($c in $leanCandidates) { if ($c -and (Test-Path $c)) { $lean = (Resolve-Path $c).Path; break } }
if (-not $lean) { Fail 'nvinfer_lean_11.dll not found (use -TrtLeanDll or env FB_TRT_LEAN_DLL)' }
if ((Split-Path $lean -Leaf) -ne 'nvinfer_lean_11.dll') { Fail ("unexpected TensorRT file name: " + $lean) }

# FFmpeg ---------------------------------------------------------------------------------------
$devFFmpeg = $false
if (-not $FFmpegDir) {
    $lgpl = Join-Path $Repo 'third_party\ffmpeg\dist\bin'
    if (Test-Path $lgpl) {
        $FFmpegDir = $lgpl
    } elseif ($AllowDevFFmpeg) {
        $root = $env:FFMPEG_DIR
        if (-not $root) { $root = 'C:\ffmpeg-8.0.1-full_build-shared' }
        $FFmpegDir = Join-Path $root 'bin'
        $devFFmpeg = $true
    } else {
        Fail 'LGPL FFmpeg not found at third_party\ffmpeg\dist\bin. Build it with third_party\ffmpeg\build_ffmpeg_lgpl.cmd (or pass -FFmpegDir; -AllowDevFFmpeg for local GPL test builds).'
    }
}
if (-not (Test-Path $FFmpegDir)) { Fail ("FFmpeg dir not found: " + $FFmpegDir) }
$FFmpegDir = (Resolve-Path $FFmpegDir).Path
if ($AllowDevFFmpeg) { $devFFmpeg = $true }
if ($devFFmpeg) {
    Warn 'Using a DEV (GPL) FFmpeg build: the result must NOT be published.'
}
Info ("ffmpeg dir  : " + $FFmpegDir)
Info ("trt lean    : " + $lean)

$ffNames = @('avcodec', 'avformat', 'avutil', 'swscale', 'swresample')
$ffRe = '^(avcodec|avformat|avutil|swscale|swresample|avfilter|avdevice|postproc)-\d+\.dll$'

# ---------------------------------------------------------------------------------------------
# Dependency closure (imports of exe / fb_native / ffmpeg DLLs)
# ---------------------------------------------------------------------------------------------
$sys32 = Join-Path $env:WINDIR 'System32'
$vcDirs = @()
$vsRoots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
foreach ($r in $vsRoots) {
    $base = Join-Path $r 'Microsoft Visual Studio'
    if (Test-Path $base) {
        $found = Get-ChildItem -Path $base -Directory -Recurse -Depth 9 -Filter 'Microsoft.VC14*.CRT' -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -match '\\x64\\' } | Sort-Object FullName -Descending
        foreach ($f in $found) { $vcDirs += $f.FullName }
    }
}
$vcRe = '^(vcruntime140|vcruntime140_1|msvcp140|msvcp140_1|msvcp140_2|concrt140|vcomp140)\.dll$'

$copy = @{}          # dest-name(lower) -> source path
$unresolved = @()

function AddFile([string]$src, [string]$destName) {
    $k = $destName.ToLower()
    if (-not $copy.ContainsKey($k)) { $copy[$k] = $src; return $true }
    return $false
}

function Resolve-Dep([string]$dll) {
    $l = $dll.ToLower()
    if ($l -match '^(api-ms-|ext-ms-)') { return $null }
    if ($l -match $ffRe) {
        $p = Join-Path $FFmpegDir $dll
        if (Test-Path $p) { return $p }
        return 'MISSING'
    }
    if ($l -match $vcRe) {
        foreach ($d in $vcDirs) { $p = Join-Path $d $dll; if (Test-Path $p) { return $p } }
        $p = Join-Path $sys32 $dll
        if (Test-Path $p) { Warn ("VC runtime " + $dll + " taken from System32 (VS redist dir not found)"); return $p }
        return 'MISSING'
    }
    $p = Join-Path $ReleaseDir $dll
    if (Test-Path $p) {
        # Never pick GPL dev FFmpeg-related or TRT builder files up by accident
        return $p
    }
    if (Test-Path (Join-Path $sys32 $dll)) { return $null }   # Windows system DLL
    return 'MISSING'
}

$queue = New-Object System.Collections.Queue
$queue.Enqueue($exe); $queue.Enqueue($fbn); $queue.Enqueue($lean)
$seen = @{}
while ($queue.Count -gt 0) {
    $f = [string]$queue.Dequeue()
    $key = $f.ToLower()
    if ($seen.ContainsKey($key)) { continue }
    $seen[$key] = $true
    foreach ($imp in [FbPe]::Imports($f)) {
        $r = Resolve-Dep $imp
        if ($null -eq $r) { continue }
        if ($r -eq 'MISSING') { $unresolved += ((Split-Path $f -Leaf) + ' -> ' + $imp); continue }
        if (AddFile $r $imp) { $queue.Enqueue($r) }
    }
}
# FFmpeg libs that the exe loads/imports are found above; make sure the five core ones are there.
foreach ($n in $ffNames) {
    $m = Get-ChildItem -Path $FFmpegDir -Filter ($n + '-*.dll') -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $m) { Fail ("FFmpeg library " + $n + "-*.dll not found in " + $FFmpegDir) }
    [void](AddFile $m.FullName $m.Name)
}
# libs that must NOT be shipped even if something imports them (dev FFmpeg has avfilter/avdevice)
foreach ($k in @($copy.Keys)) {
    if ($k -match '^(avfilter|avdevice|postproc)-') {
        # only keep if the exe really imports them; the closure above added it only in that case
        Warn ("package will contain " + $k + " because a binary imports it (not part of the minimal LGPL build: enable it with FB_FFMPEG_AVFILTER=1 or drop the rsmpeg avfilter/avdevice features)")
    }
}
if ($unresolved.Count -gt 0) {
    foreach ($u in $unresolved) { Warn ("unresolved dependency: " + $u) }
    Fail 'unresolved (non-system) DLL dependencies - the package would not start on a clean machine'
}

# forbidden files
$forbidden = '^(nvinfer_11|nvinfer_builder_resource.*|nvinfer_plugin_11|nvonnxparser_11|nvinfer_dispatch.*|nvinfer_vc_plugin.*|cublas.*|cudnn.*)\.dll$'
foreach ($k in $copy.Keys) {
    if ($k -match $forbidden) { Fail ("forbidden redistribution candidate in package: " + $k) }
}

# ---------------------------------------------------------------------------------------------
# GPL guard on FFmpeg DLLs
# ---------------------------------------------------------------------------------------------
Section 'FFmpeg licence check'
if (-not $devFFmpeg) {
    foreach ($k in $copy.Keys) {
        if ($k -match '^(avcodec|avformat|avutil|swscale|swresample)-') {
            $p = $copy[$k]
            foreach ($bad in @('--enable-gpl', '--enable-nonfree', '--enable-version3', 'x264 - core', 'x265 HEVC')) {
                if ([FbPe]::ContainsAscii($p, $bad)) { Fail ($k + ' contains "' + $bad + '" - not an LGPL build') }
            }
        }
    }
    $au = $copy.Keys | Where-Object { $_ -match '^avutil-' } | Select-Object -First 1
    $lic = [FbPe]::AsciiAfter($copy[$au], 'LGPL version', 64)
    if (-not $lic) { Fail 'avutil does not carry the "LGPL version ..." license string' }
    Info ("  avutil license string: " + $lic)
    Info '  OK: no GPL/nonfree markers in the FFmpeg DLLs'
} else {
    Warn 'licence check skipped (-AllowDevFFmpeg)'
}

# ---------------------------------------------------------------------------------------------
# Engines + manifest
# ---------------------------------------------------------------------------------------------
Section 'Engines'
if (-not (Test-Path $ManifestPath)) { Fail ("manifest not found: " + $ManifestPath) }
$manifestText = [System.IO.File]::ReadAllText($ManifestPath)
$manifest = $manifestText | ConvertFrom-Json
$engineCopies = @()
foreach ($id in $Engines) {
    $model = $manifest.models | Where-Object { $_.id -eq $id } | Select-Object -First 1
    if (-not $model) { Fail ("model '" + $id + "' is not in the manifest") }
    foreach ($arch in $Arches) {
        $eng = $model.engines | Where-Object { $_.arch -eq $arch } | Select-Object -First 1
        if (-not $eng) {
            if ($arch -eq 'ampere+') { Fail ("manifest has no ampere+ engine for " + $id) }
            Warn ("manifest has no " + $arch + " engine for " + $id + " (skipped)")
            continue
        }
        $p = Join-Path $EnginesDir $eng.file
        if (-not (Test-Path $p)) {
            if ($arch -eq 'ampere+') { Fail ("engine file missing: " + $p + "  (run the models pipeline: models\build_engines.py)") }
            Warn ("engine file missing: " + $p + " (arch " + $arch + " skipped)")
            continue
        }
        $actual = (Get-FileHash -Algorithm SHA256 -Path $p).Hash.ToLower()
        $want = ([string]$eng.sha256).ToLower()
        if ($want -match '^0+$' -or [int64]$eng.size -eq 0) {
            if ($Strict) { Fail ("manifest has a placeholder sha256 for " + $eng.file) }
            Warn ("manifest has a placeholder sha256 for " + $eng.file + " (actual " + $actual + ")")
        } elseif ($actual -ne $want) {
            Fail ("sha256 mismatch for " + $eng.file + "`n  manifest: " + $want + "`n  file    : " + $actual)
        } elseif ((Get-Item $p).Length -ne [int64]$eng.size) {
            Fail ("size mismatch for " + $eng.file)
        }
        Info ("  " + $id + " / " + $arch + " : " + $eng.file + "  " + (FmtMB (Get-Item $p).Length))
        $engineCopies += ,@($p, $eng.file)
    }
}

# ---------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------
Section ("Staging " + $PkgDir)
if (Test-Path $PkgDir) { Remove-Item -Recurse -Force $PkgDir }
New-Item -ItemType Directory -Force -Path $PkgDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PkgDir 'models') | Out-Null

Copy-Item $exe (Join-Path $PkgDir 'filming-better.exe')
Copy-Item $fbn (Join-Path $PkgDir 'fb_native.dll')
Copy-Item $lean (Join-Path $PkgDir 'nvinfer_lean_11.dll')
foreach ($k in $copy.Keys) {
    if ($k -eq 'fb_native.dll' -or $k -eq 'nvinfer_lean_11.dll') { continue }
    $leaf = Split-Path $copy[$k] -Leaf
    Copy-Item $copy[$k] (Join-Path $PkgDir $leaf)
}
foreach ($e in $engineCopies) { Copy-Item $e[0] (Join-Path $PkgDir ('models\' + $e[1])) }

# manifest: copy, fixing the OWNER placeholder of base_url when requested
$mOut = $manifestText
if ($RepoOwner) { $mOut = $mOut -replace '/OWNER/', ('/' + $RepoOwner + '/') }
if ($mOut -match '/OWNER/') { Warn 'shipped manifest.json still has the OWNER placeholder in base_url (pass -RepoOwner)' }
[System.IO.File]::WriteAllText((Join-Path $PkgDir 'models\manifest.json'), $mOut, (New-Object System.Text.UTF8Encoding($false)))

# docs
foreach ($d in @('LICENSE', 'THIRD_PARTY_NOTICES', 'README.md')) {
    $p = Join-Path $Repo $d
    if (-not (Test-Path $p)) { Fail ("missing " + $d + " in the repo root") }
    Copy-Item $p (Join-Path $PkgDir $d)
}
$rustLic = Join-Path $Repo 'THIRD_PARTY_RUST.html'
if (Test-Path $rustLic) { Copy-Item $rustLic (Join-Path $PkgDir 'THIRD_PARTY_RUST.html') }

if (-not $BuildConfig) {
    foreach ($c in @((Join-Path $Repo 'third_party\ffmpeg\dist\BUILD_CONFIG.txt'), (Join-Path $Repo 'third_party\ffmpeg\BUILD_CONFIG.txt'))) {
        if (Test-Path $c) { $BuildConfig = $c; break }
    }
}
if ($BuildConfig -and (Test-Path $BuildConfig)) {
    Copy-Item $BuildConfig (Join-Path $PkgDir 'BUILD_CONFIG.txt')
} elseif ($devFFmpeg) {
    Set-Content -Path (Join-Path $PkgDir 'BUILD_CONFIG.txt') -Encoding ASCII -Value @(
        'DEV PACKAGE - FFmpeg DLLs are a GPL development build (not redistributable).',
        'Release packages contain the LGPL build described by third_party/ffmpeg/build_ffmpeg_lgpl.sh.')
} else {
    Fail 'BUILD_CONFIG.txt (FFmpeg configure line) not found - build FFmpeg with third_party\ffmpeg\build_ffmpeg_lgpl.sh'
}
Set-Content -Path (Join-Path $PkgDir 'VERSION.txt') -Encoding ASCII -Value $Version

# unreadable-by-mistake check: forbidden names
Get-ChildItem $PkgDir -Recurse -File | ForEach-Object {
    if ($_.Name.ToLower() -match $forbidden) { Fail ("forbidden file staged: " + $_.Name) }
}

$files = Get-ChildItem $PkgDir -Recurse -File
Info ("  " + $files.Count + " files, " + (FmtMB (($files | Measure-Object Length -Sum).Sum)) + " uncompressed")
foreach ($f in ($files | Sort-Object Length -Descending)) {
    Info ("    " + (FmtMB $f.Length) + "  " + $f.FullName.Substring($PkgDir.Length + 1))
}

if ($SkipZip) { Info 'SkipZip: done.'; exit 0 }

# ---------------------------------------------------------------------------------------------
# Zip (manual, forward-slash entry names, works on .NET Framework 4.5+)
# ---------------------------------------------------------------------------------------------
Section ("Zipping " + $ZipPath)
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
$fs = [System.IO.File]::Open($ZipPath, [System.IO.FileMode]::CreateNew)
$za = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($f in ($files | Sort-Object FullName)) {
        $rel = $PkgName + '/' + $f.FullName.Substring($PkgDir.Length + 1).Replace('\', '/')
        $entry = $za.CreateEntry($rel, [System.IO.Compression.CompressionLevel]::Optimal)
        $es = $entry.Open()
        $in = [System.IO.File]::OpenRead($f.FullName)
        try { $in.CopyTo($es) } finally { $in.Dispose(); $es.Dispose() }
    }
} finally {
    $za.Dispose(); $fs.Dispose()
}
$zipBytes = (Get-Item $ZipPath).Length
$hash = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash.ToLower()
Set-Content -Path ($ZipPath + '.sha256') -Encoding ASCII -Value ($hash + '  ' + (Split-Path $ZipPath -Leaf))

# ---------------------------------------------------------------------------------------------
# Size breakdown + checks
# ---------------------------------------------------------------------------------------------
Section 'Size breakdown (compressed bytes inside the zip)'
$zr = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
$groups = @{}
$rows = @()
try {
    foreach ($e in $zr.Entries) {
        $n = $e.FullName.ToLower()
        $g = 'docs/other'
        if ($n -match '/models/.*\.engine$') { $g = 'engines' }
        elseif ($n -match '/(avcodec|avformat|avutil|swscale|swresample|avfilter|avdevice)-\d+\.dll$') { $g = 'ffmpeg dlls' }
        elseif ($n -match 'nvinfer') { $g = 'tensorrt lean' }
        elseif ($n -match '/(filming-better\.exe|fb_native\.dll)$') { $g = 'app (exe + fb_native)' }
        elseif ($n -match '\.dll$') { $g = 'other dlls' }
        if (-not $groups.ContainsKey($g)) { $groups[$g] = @(0, 0) }
        $groups[$g][0] += $e.CompressedLength
        $groups[$g][1] += $e.Length
        $rows += New-Object psobject -Property @{ Name = $e.FullName; Comp = $e.CompressedLength; Raw = $e.Length }
    }
} finally { $zr.Dispose() }
foreach ($r in ($rows | Sort-Object Comp -Descending | Select-Object -First 12)) {
    Info ("  " + (FmtMB $r.Comp) + " (raw " + (FmtMB $r.Raw) + ")  " + $r.Name)
}
Info ''
foreach ($g in ($groups.Keys | Sort-Object)) {
    Info ("  {0,-24} {1}  (raw {2})" -f $g, (FmtMB $groups[$g][0]), (FmtMB $groups[$g][1]))
}
Info ''
Info ("  ZIP total: " + (FmtMB $zipBytes) + "   limit " + $MaxZipMB + " MB")
Info ("  sha256   : " + $hash)

$rc = Test-Sizes
Info ''
if ($rc -eq 0) { Info ("DONE: " + $ZipPath) }
exit $rc
