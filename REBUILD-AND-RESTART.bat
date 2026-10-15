@echo off
rem XeStrata-win REBUILD-AND-RESTART: incremental ninja build of strata.exe, then the server restarts.
rem The running engine locks build-rel\strata.exe, so the link can only happen after the stop - that is
rem what this bat sequences. For a plain restart (no build) use START-HERE.bat.
rem The first start after an arena-mmap change writes experts.bin and still runs pinned; the NEXT start
rem maps it (log: "expert arena: mapped read-only from ...").

rem auto-elevate (same as START-HERE): PCIe counters need it; the build does not care.
fltmc >nul 2>&1 || (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

setlocal
title XeStrata rebuild + restart
cd /d "%~dp0"
set XE=%~dp0
set PYVENV=%~dp0.venv

echo [1/5] stopping the running server (the link needs the exe unlocked)...
rem The rebuild must stop the server that locks strata.exe. But an accidental second launch must
rem not kill a server that is not ours: if a server is running, wait up to 2 minutes for it to
rem close on its own; if it is still up, abort before the build - nothing is ever killed.
set /a TRIES=0
set PSCHECK=%TEMP%\xestrata_conflict_check.ps1
> "%PSCHECK%" echo $p = Get-CimInstance Win32_Process ^| Where-Object { ($_.Name -match 'python' -and $_.CommandLine -match 'server\.py') -or ($_.Name -eq 'strata.exe') -or ($_.Name -match 'powershell' -and $_.CommandLine -match 'keepwarm') }
>> "%PSCHECK%" echo if ($p) { $p ^| ForEach-Object { '     running: PID ' + $_.ProcessId + '  ' + $_.Name }; 'CONFLICT=YES' } else { 'CONFLICT=NO' }
:checkconflict
set CONFLICT=NO
for /f "usebackq tokens=2 delims==" %%c in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%PSCHECK%"`) do ( if "%%c"=="YES" ( set CONFLICT=YES ) )
if "%CONFLICT%"=="NO" ( del "%PSCHECK%" >nul 2>&1 & goto conflictclear )
set /a TRIES+=1
if %TRIES% GTR 24 ( echo     a Strata server is still running after 2 minutes - not rebuilding, so nothing gets killed. & echo     close the server's window first, then run this again. & del "%PSCHECK%" >nul 2>&1 & pause & exit /b 1 )
echo     a Strata server is running - waiting for it to close on its own (attempt %TRIES%/24, 5 s each)...
timeout /t 5 /nobreak >nul
goto checkconflict
:conflictclear

echo [2/5] building strata.exe (incremental ninja)...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
cd /d "%XE%build-rel"
ninja strata
if errorlevel 1 ( echo     BUILD FAILED - the server was NOT restarted. Read the lines above. & pause & exit /b 1 )
if not exist libblosc2.dll ( echo     libblosc2.dll missing in build-rel - the server would fail to start. & pause & exit /b 1 )

echo [3/5] waiting for free RAM (needs 45 GB+; the arena is 46.84 GiB)...
:waitmem
for /f "usebackq delims=" %%m in (`powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"`) do set FREEGB=%%m
echo     free RAM: %FREEGB% GB
if %FREEGB% LSS 45 ( echo     waiting 10 s... & timeout /t 10 /nobreak >nul & goto waitmem )

echo [4/5] starting the XeStrata server on :8080 (engine load takes ~1-2 min; the first arena-mmap start
echo       also writes experts.bin, a few minutes more)...
echo     this window stays open while the server runs; closing it stops the server.
set ZES_ENABLE_SYSMAN=1
set STRATA_SKIP_DEVICE_TUNE=1
set STRATA_ARENA_NO_REGISTER=1
rem keepwarm runs inside the server now (config "keepwarm": true) - no separate window.
"%PYVENV%\Scripts\python.exe" "%XE%serve\server.py" --engine strata --config "%XE%strata-xe-iq3_s.json" --port 8080
echo [5/5] the server exited.
pause
