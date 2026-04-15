# Virtual Environment Quick Commands

## ✅ Status

Your `.venv` virtual environment is **READY** at:
```
c:\github_projects\data_science\data_science\GenAI_LLM\langgraph_deeplearningai_course\shopping_assistant\.venv
```

**Installed packages:**
- langgraph ✓
- langchain ✓
- duckduckgo-search ✓

---

## 🚀 Quick Commands

### Activate Virtual Environment (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### Activate Virtual Environment (Command Prompt)
```cmd
.venv\Scripts\activate.bat
```

### Run the Demo
```bash
# With venv activated:
python demo.py

# Or directly:
.\.venv\Scripts\python.exe demo.py
```

### Show Diagrams
```bash
.\.venv\Scripts\python.exe visualize.py
```

### Run Individual Node Test
```bash
.\.venv\Scripts\python.exe -c "from nodes.input_node import input_node; print('✅ Input node works!')"
```

### Install a New Package
```bash
# With venv activated:
pip install package-name

# Or directly:
.\.venv\Scripts\python.exe -m pip install package-name
```

### Check Installed Packages
```bash
.\.venv\Scripts\python.exe -m pip list
```

### Deactivate Virtual Environment
```bash
deactivate
```

---

## 📂 Directory Structure

```
shopping_assistant/
├── .venv/                           ← Virtual environment (isolated Python)
│   ├── Scripts/                     ← Python executable & pip
│   ├── Lib/site-packages/           ← Installed packages
│   ├── share/                       ← Documentation
│   └── pyvenv.cfg                   ← Config file
│
├── data/
│   ├── books_catalog.py
│   └── web_search.py
├── nodes/
│   ├── input_node.py
│   ├── search_node.py
│   ├── filter_node.py
│   ├── summarizer_node.py
│   └── output_node.py
│
├── demo.py
├── workflow.py
├── requirements.txt                 ← List of dependencies
├── VENV_SETUP.md                    ← Setup instructions
└── VENV_COMMANDS.md                 ← This file!
```

---

## 🔍 Verify Installation

Check that everything works:

```bash
.\.venv\Scripts\python.exe -c "
import langgraph
import langchain
import duckduckgo_search
print('✅ All packages installed!')
"
```

---

## 📝 Typical Workflow

1. **Navigate to project:**
   ```bash
   cd shopping_assistant
   ```

2. **Activate environment:**
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

3. **Run scripts:**
   ```bash
   python demo.py
   ```

4. **Deactivate when done:**
   ```bash
   deactivate
   ```

---

## ❓ Troubleshooting

**"Python is not recognized"**
- Use full path: `.\.venv\Scripts\python.exe`

**"Module not found"**
- Ensure venv is activated
- Check package is in requirements.txt
- Run: `pip install -r requirements.txt`

**"Permission denied on Activate.ps1"**
- Use Command Prompt instead: `venv\Scripts\activate.bat`
- Or run PowerShell as Administrator

**"Want to delete and recreate venv?"**
```bash
rmdir /s /q .venv
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

✨ **Ready to code!** Run `python demo.py` to get started! 🚀
