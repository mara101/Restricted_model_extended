@echo off
title Droplet alpha-r scan
REM Uses a separate conda env "tenpy_droplets" with TeNPy >= 1.1 (needed for dipole conservation).
REM Your existing tenpy_env is NOT modified. Re-launching resumes the scan.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
cd /d "%USERPROFILE%\Documents\Quantum Engineering\Quantum Engineering\Research\Quantum-Many-Body-Systems\Codes\Restricted_model_extended"
if not exist droplet_alpha_r_data mkdir droplet_alpha_r_data
set LOG=droplet_alpha_r_data\scan_log.txt
call conda activate tenpy_droplets 2>nul
if errorlevel 1 (
  echo Creating conda environment tenpy_droplets ^(first time only, a few minutes^)...
  echo Creating env tenpy_droplets %DATE% %TIME% >> %LOG%
  call conda create -y -n tenpy_droplets python=3.11 >> %LOG% 2>&1
  call conda activate tenpy_droplets
  python -m pip install "physics-tenpy>=1.1" numpy scipy pandas matplotlib psutil >> %LOG% 2>&1
)
python -c "import tenpy; assert int(tenpy.__version__.split(chr(46))[0])>=1, tenpy.__version__; print(chr(84)+chr(101)+chr(78)+chr(80)+chr(121), tenpy.__version__)"
if errorlevel 1 (
  echo TeNPy in tenpy_droplets is missing or too old - see %LOG%
  pause
  exit /b 1
)
echo.
echo Scan running. Progress is written to:
echo   %CD%\%LOG%
echo You can minimise this window. Do NOT close it until it says "All done".
echo Started %DATE% %TIME% >> %LOG%
python -u droplet_alpha_r_scan.py --out droplet_alpha_r_data >> %LOG% 2>&1
if errorlevel 1 (
  echo The scan stopped with an error - see %LOG%
  pause
  exit /b 1
)
echo Scan finished %DATE% %TIME% >> %LOG%
python droplet_phase_diagram.py --out droplet_alpha_r_data >> %LOG% 2>&1
echo All done. Results in droplet_alpha_r_data. You can close this window.
pause
