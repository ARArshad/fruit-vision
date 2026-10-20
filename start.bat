@echo off
REM FruitVision launcher for Windows -- just double-click this file.
setlocal
cd /d "%~dp0"

py -3 --version >nul 2>&1
if %errorlevel%==0 (
    py -3 run.py
    goto done
)

python --version >nul 2>&1
if %errorlevel%==0 (
    python run.py
    goto done
)

echo.
echo   Python was not found on this computer.
echo.
echo   Install it for free from https://www.python.org/downloads/
echo   During installation, tick "Add Python to PATH" on the first screen.
echo   Then double-click this file again.
echo.

:done
echo.
pause
