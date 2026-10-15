@echo off
rem XeStrata-win CONFIGURE-AND-BUILD: first full build (build-rel does not exist yet).
rem Needs oneAPI icx; CMake+Ninja come from the venv (activate it first or SETUP.bat calls this after venv).
rem blosc2 headers+import lib are bundled in third_party\blosc2; libblosc2.dll is already in build-rel.
setlocal
title XeStrata-win configure + build
cd /d "%~dp0"

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" >nul 2>&1

cmake -G Ninja -B build-rel -S . ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" ^
  -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe" ^
  -DSTRATA_ENABLE_XE=ON ^
  -DSTRATA_NATIVE_EXPERTS=ON ^
  -DSTRATA_LICENSE=contrib-icpx ^
  -DSTRATA_PORTABLE=ON ^
  -DSTRATA_ONEMKL=OFF ^
  -DSTRATA_WERROR=OFF ^
  -DSTRATA_BUILD_TESTS=OFF ^
  -DSTRATA_BLOSC2_INCLUDE="%~dp0third_party\blosc2\include" ^
  -DSTRATA_BLOSC2_LIB="%~dp0third_party\blosc2\blosc2.lib"
if errorlevel 1 ( echo CONFIGURE FAILED & pause & exit /b 1 )

cmake --build build-rel --target strata
if errorlevel 1 ( echo BUILD FAILED & pause & exit /b 1 )

copy /Y "third_party\blosc2\libblosc2.dll" build-rel\ >nul 2>&1
if not exist "build-rel\libblosc2.dll" ( echo     copy libblosc2.dll into build-rel - it is in third_party\blosc2\ & pause & exit /b 1 )
echo BUILD OK: build-rel\strata.exe
pause
