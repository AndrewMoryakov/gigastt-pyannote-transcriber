@echo off
rem Double-click launcher for the local web UI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\ui.ps1" %*
if errorlevel 1 pause
