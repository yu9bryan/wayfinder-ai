@echo off
rem Start the recorder page (mic and folder picker need localhost)
cd /d "%~dp0"
start "" http://localhost:8765/recorder.html
python -m http.server 8765 --bind 127.0.0.1
