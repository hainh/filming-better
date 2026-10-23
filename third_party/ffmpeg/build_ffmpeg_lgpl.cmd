@echo off
rem Launches build_ffmpeg_lgpl.sh inside MSYS2 with the MSVC x64 environment (cl.exe, link.exe, lib.exe).
rem Usage: third_party\ffmpeg\build_ffmpeg_lgpl.cmd [--dry-run] [--jobs N] ...
rem Requires: MSYS2 in C:\msys64 (override with MSYS2_ROOT), VS 2022 (override with VCVARS).
setlocal
if not defined MSYS2_ROOT set "MSYS2_ROOT=C:\msys64"
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%MSYS2_ROOT%\usr\bin\bash.exe" (echo MSYS2 not found at %MSYS2_ROOT% & exit /b 1)
if not exist "%VCVARS%" (echo vcvars64.bat not found: %VCVARS% & exit /b 1)
call "%VCVARS%" >nul || exit /b 1
set "MSYSTEM=MSYS"
set "CHERE_INVOKING=1"
set "MSYS2_PATH_TYPE=inherit"
"%MSYS2_ROOT%\usr\bin\bash.exe" -lc "exec bash \"$(cygpath -u '%~dp0build_ffmpeg_lgpl.sh')\" %*"
endlocal
