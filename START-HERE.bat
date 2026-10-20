@echo off
rem XeStrata-win START-HERE: the B70 engine as the :8080 OpenAI server (P8+A+B+C production config).
rem The window stays open while the server runs; close it to stop. To go back to the 4070 (Strata-main),
rem run Desktop\RESTORE-Strata-main.bat.

rem auto-elevate: the Monitor tab's PCIe throughput counters (zesDevicePciGetStats) need an elevated process;
rem every other reading works either way. Say Yes to the one UAC prompt this shows.
fltmc >nul 2>&1 || (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

setlocal
title XeStrata server :8080 (IQ3_S) - close this window to stop
cd /d "%~dp0"
set XE=%~dp0
set PYVENV=%~dp0.venv

rem SETUP gate: on a fresh machine there is no serve config yet (SETUP.bat writes it as its last
rem step), so SETUP has not been run. Run SETUP.bat right here, then continue with the normal
rem startup below. The XE_SETUP_FROM_STARTHERE marker tells SETUP it was called from here: it then
rem just returns instead of chaining back into START-HERE (which would loop).
if not exist "%XE%strata-xe-iq3_s.json" (
    echo     no serve config yet - this tree has not been set up. Running SETUP.bat now...
    echo.
    set XE_SETUP_FROM_STARTHERE=1
    call "%XE%SETUP.bat"
    if errorlevel 1 ( echo     SETUP did not finish - not starting. Fix it and run START-HERE.bat again. & pause & exit /b 1 )
    if not exist "%XE%strata-xe-iq3_s.json" ( echo     SETUP finished but the config is still missing - not starting. & pause & exit /b 1 )
    echo.
    echo     SETUP done - continuing with the normal startup.
    echo.
)

echo [1/5] checking for another running Strata server (this build never kills another server)...
rem A second launch must NOT kill the server that is already serving :8080 - that server could be
rem this same tree, restarted by accident. So: detect any running server (any tree), wait for it
rem to close on its own, and if it is still up after 2 minutes, abort without starting.
rem Nothing is ever killed here.
set /a TRIES=0
set PSCHECK=%TEMP%\xestrata_conflict_check.ps1
> "%PSCHECK%" echo $self = $env:XE_STARTHERE_SELF
>> "%PSCHECK%" echo $p = Get-CimInstance Win32_Process ^| Where-Object { ($_.Name -eq 'cmd.exe' -and $_.CommandLine -imatch 'START-HERE' -and $_.CommandLine -inotlike ('*' + $self + '*')) -or ($_.Name -match 'python' -and $_.CommandLine -match 'server\.py') -or ($_.Name -eq 'strata.exe') -or ($_.Name -match 'powershell' -and $_.CommandLine -imatch 'keepwarm') }
>> "%PSCHECK%" echo if ($p) { $p ^| ForEach-Object { '     running: PID ' + $_.ProcessId + '  ' + $_.Name }; 'CONFLICT=YES' } else { 'CONFLICT=NO' }
set XE_STARTHERE_SELF=%~f0
:checkconflict
set CONFLICT=NO
for /f "usebackq tokens=2 delims==" %%c in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%PSCHECK%"`) do ( if "%%c"=="YES" ( set CONFLICT=YES ) )
if "%CONFLICT%"=="NO" ( del "%PSCHECK%" >nul 2>&1 & goto conflictclear )
set /a TRIES+=1
if %TRIES% GTR 24 ( echo     another Strata server is still running after 2 minutes - not starting, so nothing gets killed. & echo     close the other server's window first, then run this again. & echo     (if a strata.exe lingers with no window, kill it by hand: taskkill /F /IM strata.exe) & del "%PSCHECK%" >nul 2>&1 & pause & exit /b 1 )
echo     another Strata server is running - waiting for it to close on its own (attempt %TRIES%/24, 5 s each)...
timeout /t 5 /nobreak >nul
goto checkconflict
:conflictclear

echo [2/5] waiting for free RAM (needs 45 GB+; the arena is 46.84 GiB)...
:waitmem
for /f "usebackq delims=" %%m in (`powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"`) do set FREEGB=%%m
echo     free RAM: %FREEGB% GB
if %FREEGB% LSS 45 ( echo     waiting 10 s... & timeout /t 10 /nobreak >nul & goto waitmem )

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1
set ZES_ENABLE_SYSMAN=1
set STRATA_SKIP_DEVICE_TUNE=1
set STRATA_ARENA_NO_REGISTER=1

rem --- show what is about to start: engine version, git HEAD, startup options -------------
if not exist "%XE%strata-xe-iq3_s.json" ( echo     no config yet - run SETUP.bat first. & pause & exit /b 1 )
echo [3/5] engine version and startup options...
for /f "usebackq delims=" %%v in (`"%XE%build-rel\strata.exe" --version 2^>nul`) do echo     engine:  %%v
for /f "usebackq delims=" %%g in (`git -C "%XE%." rev-parse --short HEAD 2^>nul`) do echo     git:     %~nx0 @ %%g
powershell -NoProfile -Command "$c=Get-Content '%XE%strata-xe-iq3_s.json' -Raw|ConvertFrom-Json; '  pack:    ' + $c.args[1]; '  model:   ' + $c.model_name + '  port ' + $c.port; '  ctx:     ' + $c.args[$c.args.IndexOf('--max-context')+1] + ' tokens, kv ' + $c.args[$c.args.IndexOf('--kv')+1]; '  env:     ' + (($c.env.PSObject.Properties|ForEach-Object{$_.Name+'='+$_.Value}) -join '  ')"
echo.

echo [4/5] starting the XeStrata server on :8080 (engine load takes ~1-2 min)...
echo     this window stays open while the server runs; closing it stops the server.
rem keepwarm runs inside the server now (config "keepwarm": true) - no separate window.
"%PYVENV%\Scripts\python.exe" "%XE%serve\server.py" --engine strata --config "%XE%strata-xe-iq3_s.json" --port 8080
echo [5/5] the server exited.
pause
