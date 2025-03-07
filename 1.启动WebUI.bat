@echo off
chcp 65001 >nul

SET CONDA_PATH=.\Miniconda3

REM 激活base环境
CALL %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%

SET FFMPEG_PATH=%cd%\ffmpeg\bin
SET PATH=%FFMPEG_PATH%;%PATH%

set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\hf_download
set TORCH_CUDA_CACHE_PATH=%CD%\hf_download
set XFORMERS_FORCE_DISABLE_TRITON=1


python app.py

pause
