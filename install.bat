@echo off
REM FaceSwap Application Installation Script for Windows

echo 🚀 FaceSwap Application Installation
echo ====================================

REM Check if Python is installed
echo 🐍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Found Python %python_version%

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ℹ️  Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version by default)
echo 🔥 Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM Install other requirements
echo 📋 Installing requirements...
pip install -r requirements.txt

REM Install the package in development mode
echo 📦 Installing FaceSwap package...
pip install -e .

REM Run installation test
echo 🧪 Running installation test...
python test_installation.py

echo.
echo 🎉 Installation completed successfully!
echo.
echo 📖 Usage:
echo    venv\Scripts\activate.bat  REM Activate virtual environment
echo    python faceswap.py sample_data\video\sample.mp4 sample_data\images\
echo.
echo 📚 For more information, see README.md
echo.
pause