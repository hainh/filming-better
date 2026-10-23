@echo off
rem Dev environment for filming-better. Usage:  tools\devenv.cmd <command...>
rem   (or `call tools\devenv.cmd` with no args to just set up the current cmd shell)
set "FB_ROOT=%~dp0.."
for %%I in ("%FB_ROOT%") do set "FB_ROOT=%%~fI"
if not defined VSCMD_VER call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=%FB_ROOT%\.tools\cuda"
set "CUDAToolkit_ROOT=%CUDA_PATH%"
set "TENSORRT_INCLUDE=%FB_ROOT%\.tools\tensorrt\include"
set "SP=%FB_ROOT%\.venv\Lib\site-packages"
set "TRT_LIBS=%SP%\tensorrt_libs"
set "TRT_LEAN_LIBS=%SP%\tensorrt_lean_libs"
set "FB_TRT_LEAN_DLL=%SP%\tensorrt_lean_libs\nvinfer_lean_11.dll"
set "LIBCLANG_PATH=%SP%\clang\native"
set "FFMPEG_DIR=C:\ffmpeg-8.0.1-full_build-shared"
set "FFMPEG_INCLUDE_DIR=%FFMPEG_DIR%\include"
set "FFMPEG_LIBS_DIR=%FFMPEG_DIR%\lib"
set "PATH=%CUDA_PATH%\bin;%FB_ROOT%\.venv\Scripts;%SP%\cmake\data\bin;%TRT_LIBS%;%TRT_LEAN_LIBS%;%FFMPEG_DIR%\bin;%PATH%"
if "%~1"=="" goto :eof
%*
