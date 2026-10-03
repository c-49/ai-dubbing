@echo off
rem Starts the review app (runs setup first if this machine has not been set up).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Dubber.ps1" %*
if errorlevel 1 pause
