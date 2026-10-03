@echo off
rem One-time setup. Options: -Cpu  -NoClone  -SkipOllama
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (
  echo.
  echo Setup did not finish. Fix the problem above and run setup.bat again - it picks up where it left off.
)
pause
