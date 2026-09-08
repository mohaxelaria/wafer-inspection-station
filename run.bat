@echo off
REM Launcher for Anaconda. Creates the conda env on first run.
call conda activate wafer-inspection 2>nul
if errorlevel 1 (
    echo Creating conda environment "wafer-inspection"...
    call conda env create -f environment.yml
    call conda activate wafer-inspection
)
python -m uvicorn backend.main:app --reload
