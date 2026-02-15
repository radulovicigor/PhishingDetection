@echo off
title Phish Detector
cd /d "%~dp0"

echo.
echo  ========================================
echo     PHISH DETECTOR - Pokretanje
echo  ========================================
echo.

:: Provjeri da li Python postoji
where python >nul 2>&1
if errorlevel 1 (
    echo [GRESKA] Python nije instaliran!
    echo          Preuzmi sa https://python.org
    pause
    exit /b
)

:: Kreiraj venv ako ne postoji
if not exist ".venv\Scripts\python.exe" (
    echo [*] Prva instalacija - kreiranje okruzenja...
    python -m venv .venv
    if errorlevel 1 (
        echo [GRESKA] Ne mogu kreirati okruzenje!
        pause
        exit /b
    )
    
    echo [*] Instalacija biblioteka (1-2 minuta)...
    .venv\Scripts\pip.exe install -r requirements.txt
    if errorlevel 1 (
        echo [GRESKA] Instalacija nije uspjela!
        pause
        exit /b
    )
    echo.
    echo [*] Instalacija zavrsena!
    echo.
)

echo [*] Pokrecem aplikaciju...
echo.
echo     Otvori u browseru: http://localhost:8000
echo.
echo     Za zatvaranje - zatvori ovaj prozor
echo.

:: Otvori browser
start http://localhost:8000

:: Pokreni server koristeci Python iz venv-a
.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000

pause
