@echo off
echo ==========================================
echo   FMCG Coder Local Setup (Ollama + Crawl4AI)
echo ==========================================

:: 1. Create Virtual Environment
echo [1/5] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo FAILED: Could not create virtual environment.
    pause
    exit /b
)

:: 2. Activate Environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate

:: 3. Install Requirements
echo [3/5] Installing Python libraries...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo FAILED: Pip installation failed.
    pause
    exit /b
)

:: 4. Initialize Playwright (For Crawl4AI)
echo [4/5] Installing Chromium for Crawl4AI...
python -m playwright install --with-deps chromium

:: 5. Initialize Crawl4AI
echo [5/5] Running Crawl4AI diagnostics...
crawl4ai-setup

echo ==========================================
echo   SETUP COMPLETE!
echo   To start, run: venv\Scripts\activate
echo   Then: python main.py
echo ==========================================
pause