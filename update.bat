@echo off
rem Pulls new code (if this is a git checkout) and re-syncs the environment.
rem Never touches models\, work\ or your show data.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
pause
