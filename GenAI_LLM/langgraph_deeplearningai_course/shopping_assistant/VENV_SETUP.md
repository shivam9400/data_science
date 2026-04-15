# Virtual Environment Setup

Your `.venv` virtual environment is now ready! 🚀

## 📁 Location

```
shopping_assistant/
└── .venv/                    ← Your virtual environment
    ├── Scripts/              ← Executable files (Python, pip, etc.)
    ├── Lib/                  ← Installed packages
    └── pyvenv.cfg            ← Configuration
```

## ✅ Installed Packages

- **langgraph** (1.1.6) - Graph orchestration framework
- **langchain** (1.2.15) - LLM utilities
- **duckduckgo-search** (8.1.1) - Open-source web search
- Plus all dependencies

## 🚀 How to Use

### Windows (PowerShell)

**Activate the virtual environment:**
```powershell
.\.venv\Scripts\Activate.ps1
```

(If you get a security error, you may need to enable script execution or use cmd.exe)

### Windows (Command Prompt)

**Activate the virtual environment:**
```cmd
.venv\Scripts\activate.bat
```

### Run Python Scripts

**After activation:**
```bash
python demo.py
python workflow.py
```

**Or without activation (using full path):**
```bash
.\.venv\Scripts\python.exe demo.py
```

## 📦 Install Additional Packages

If you need to install more packages:

```bash
# After activation
pip install package-name

# Or without activation
.\.venv\Scripts\python.exe -m pip install package-name
```

## 🔄 Update Requirements

If you add new packages, update `requirements.txt`:

```bash
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

## 🗑️ Deactivate Virtual Environment

```bash
deactivate
```

## 🔁 Recreate Virtual Environment

If needed, you can delete `.venv` and recreate it:

```bash
# Delete
rmdir /s /q .venv

# Recreate
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## ✨ You're All Set!

Your project is now running in an isolated Python environment. Nothing will interfere with your system Python or other projects.

**Next:** Run `python demo.py` to test the shopping assistant! 🎉
