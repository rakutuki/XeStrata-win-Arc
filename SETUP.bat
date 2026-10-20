@echo off
rem XeStrata-win SETUP: check this PC, make .venv, build if needed, write the serve config.
rem Run this once on a new machine (double-click); when it finishes it starts the server itself
rem (it calls START-HERE.bat). START-HERE.bat also runs SETUP first if the tree is not set up yet.
setlocal EnableDelayedExpansion
title XeStrata-win SETUP
cd /d "%~dp0"
set "ONEAPI=C:\Program Files (x86)\Intel\oneAPI"

echo == XeStrata-win SETUP ==
echo.

echo [1/6] GPU...
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' } | ForEach-Object { '  GPU: ' + $_.Name }" 2>nul
powershell -NoProfile -Command "if (-not (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'Intel' })) { '  NO Intel GPU found - XeStrata needs an Intel Arc (B70 verified; B580/B570/B60 untested)' }" 2>nul

echo [2/6] RAM (the expert arena is 46.84 GiB pinned)...
for /f "usebackq delims=" %%m in (`powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize/1MB)"`) do set TOTALGB=%%m
echo     total RAM: %TOTALGB% GB
if %TOTALGB% LSS 56 ( echo     WARNING: under 56 GB total - the engine needs ~60 GB free-class machines; it may refuse to start. & pause )

echo [3/6] oneAPI runtime...
if not exist "%ONEAPI%\setvars.bat" ( call :install_oneapi || exit /b 1 )

echo [4/6] python venv...
rem a .venv built with a Python that was since uninstalled still has python.exe but dies on start
rem ("did not find executable at ..."): test-run it, and rebuild the venv when it is broken.
if not exist ".venv\Scripts\python.exe" goto :makevenv
".venv\Scripts\python.exe" -c "print(1)" >nul 2>nul && goto :venvok
echo     the existing .venv is broken - its base Python was uninstalled. Rebuilding it...
rmdir /s /q .venv
:makevenv
where py >nul 2>nul || where python >nul 2>nul || ( call :install_python || exit /b 1 )
py -3 -m venv .venv 2>nul || python -m venv .venv || ( echo     still no Python - install Python 3.10+ yourself ^(python.org, or: winget install Python.Python.3.13^). & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo     pip install failed & pause & exit /b 1 )
:venvok
echo     venv: .venv

echo [5/6] engine...
if exist "build-rel\strata.exe" (
    echo     build-rel\strata.exe present - no build needed.
) else (
    echo     build-rel\strata.exe missing - full build ^(needs oneAPI icx; ~1 h^).
    call CONFIGURE-AND-BUILD.bat
    if errorlevel 1 exit /b 1
)

echo [6/6] model files + serve config...
set "DD=%USERPROFILE%\Desktop\XeStrata-data"
set /p "DD=Where are packs\ models\ mtp\? [default %DD%]: "
".venv\Scripts\python.exe" tools\setup_choices.py
if errorlevel 1 ( pause & exit /b 1 )
if not exist "%DD%\packs" ( call :fetch_models "%DD%" || exit /b 1 )
".venv\Scripts\python.exe" tools\setup_config.py --data-dir "%DD%"
if errorlevel 1 ( pause & exit /b 1 )

echo.
if defined XE_SETUP_FROM_STARTHERE (
    echo SETUP done - START-HERE.bat continues the startup from where it paused.
    exit /b 0
)
echo SETUP done. Starting the server now, exactly as START-HERE.bat does...
echo.
call "%~dp0START-HERE.bat"
exit /b 0

rem ---- the data folder is empty: fetch the model files from Hugging Face (asks first; ~60-90 GB).
:fetch_models
echo     no packs\ folder in %~1 - the model files (~60-90 GB) are not there yet.
echo     SETUP can download them from Hugging Face now: pick the model by this PC's RAM,
echo     the shards resume if the line drops, and the pack + MTP layer are built after.
set "YN=n"
set /p "YN=Download the model files now? [y/N]: "
if /i not "!YN!"=="y" ( echo     skipped - put packs\ models\ mtp\ under %~1 yourself ^(README-WIN: Model files^), then run SETUP.bat again. & exit /b 1 )
".venv\Scripts\python.exe" tools\setup_models.py --data-dir "%~1"
if errorlevel 1 exit /b 1
exit /b 0

rem ---- this PC has no oneAPI: winget installs the oneAPI Toolkit 2026.x (asks first).
rem The bundled strata.exe imports sycl9.dll, which only the 2026.x runtime ships; winget's older
rem "Base Toolkit" 2025.1 ships the sycl8-era runtime and will NOT run it, so the Toolkit is what we offer.
:install_oneapi
echo     oneAPI not found ^(setvars.bat missing^). The bundled engine needs the 2026.x oneAPI runtime.
echo     winget can install the Intel oneAPI Toolkit now ^(several GB, takes a while^).
set "YN=y"
set /p "YN=Install Intel oneAPI Toolkit via winget? [Y/n]: "
if /i "!YN!"=="n" ( echo     skipped - install it yourself and run SETUP.bat again: & echo       https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html & exit /b 1 )
winget install -e --id Intel.OneAPI.Toolkit --source winget --accept-package-agreements --accept-source-agreements
if not exist "%ONEAPI%\setvars.bat" ( echo     oneAPI still not found after winget ^(or winget failed^) - install it yourself: & echo       https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html & exit /b 1 )
echo     oneAPI installed: %ONEAPI%
exit /b 0

rem ---- this PC has no Python: winget installs Python 3.13 (asks first; nothing else is installed).
:install_python
echo     Python not found. winget can install Python 3.13 now (about 100 MB).
set "YN=y"
set /p "YN=Install Python 3.13 via winget? [Y/n]: "
if /i "!YN!"=="n" ( echo     skipped - install Python 3.10+ yourself and run SETUP.bat again. & exit /b 1 )
winget install -e --id Python.Python.3.13 --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 ( echo     winget could not install Python - install it yourself ^(python.org^) and run SETUP.bat again. & exit /b 1 )
rem the installer writes PATH, but this window does not see it yet: point at the default install folders.
set "PATH=%LOCALAPPDATA%\Programs\Python\Launcher;%LOCALAPPDATA%\Programs\Python\Python313;%LOCALAPPDATA%\Programs\Python\Python313\Scripts;%PATH%"
echo     Python installed; continuing with the py launcher.
exit /b 0
