@echo off
REM Convenience launcher for Windows.
if not exist .venv (
    python -m venv .venv
    .venv\Scripts\python -m pip install -q -r requirements.txt
)
.venv\Scripts\python -m uvicorn backend.main:app --reload
