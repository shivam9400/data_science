# 🚀 QUICK START - WEB SEARCH EDITION

Get the shopping assistant with web search up and running in minutes

## **1️⃣ INSTALL**

```bash
cd shopping_assistant
pip install -r requirements.txt
```

## **2️⃣ TEST WITH DEMOS**

```bash
python demo.py
```

## **3️⃣ TRY WEB SEARCH**

```python
from workflow import run_shopping_assistant

# Search the web!
run_shopping_assistant(
    query="python programming",
    search_source="web"
)
```

## **4️⃣ EXPLORE OPTIONS**

```python
# Catalog only (fast, demo mode)
run_shopping_assistant(query="cooking", search_source="catalog")

# Web only (real results, unlimited)
run_shopping_assistant(query="python books", search_source="web")

# Both together (comprehensive)
run_shopping_assistant(query="cooking", search_source="both")
```

## 📖 LEARN MORE

### Detailed web search info:
```bash
python WEB_SEARCH_GUIDE.py
```

### What changed:
```bash
python WEB_SEARCH_INTEGRATION_SUMMARY.py
```

### Full documentation:
See **README.md** ("Web Search Feature" section)

## 📁 New Files

- **data/web_search.py** - DuckDuckGo integration
- **WEB_SEARCH_GUIDE.md** - Full guide
- **WEB_SEARCH_INTEGRATION_SUMMARY.md** - What changed

## 🔄 Modified Files

- **search_node.py** - Supports catalog/web/both
- **filter_node.py** - Handles mixed results
- **summarizer_node.py** - Uses snippets
- **output_node.py** - Shows links
- **workflow.py** - Added search_source parameter
- **demo.py** - 4 demos including web search
- **requirements.txt** - Added duckduckgo-search

## 💡 THREE DEMO SCENARIOS

Run: `python demo.py`

### Demo 1: Catalog Search
- Search: Local 6 books
- Speed: ⚡ Instant
- Has: Price + Rating

### Demo 2: Web Search 🌐
- Search: Entire internet
- Speed: 🐌 2-3 seconds
- Has: Links + Snippets

### Demo 3: Hybrid Search 🌐
- Search: Catalog + Web
- Speed: 🔄 Medium
- Has: Best of both

### Demo 4: Budget Filter
- Search: Catalog <$25
- Speed: ⚡ Instant
- Has: Cheap + Good rating

## 🎯 USE IN YOUR PRESENTATION

```
"Let me show you how the same workflow works with different sources:

1. First, catalog only (fast, demo)
   python demo.py  [See Demo 1]

2. Now let's search the web (real results!)
   python demo.py  [See Demo 2]

3. Finally, combining both (best approach)
   python demo.py  [See Demo 3]

See how the structure stays the same? Only the data source changes!"
```

## ⚙️ KEY FACTS

- ✓ Open-source (DuckDuckGo)
- ✓ No API keys needed
- ✓ No authentication required
- ✓ Privacy-friendly
- ✓ Easy to customize
- ✓ Works offline (catalog mode)
- ✓ Real-time (web mode)
- ✓ Modular (easy to swap engines)

## 🔧 CUSTOMIZE

### Use different search engine:
Edit: **data/web_search.py**

### Adjust number of results:
Edit: **search_node.py** (line with max_results)

### Add pricing extraction:
Edit: **summarizer_node.py**

### Add result filtering:
Edit: **filter_node.py**

## ❓ TROUBLESHOOTING

**Q: "Web search is slow"**
A: Normal! First request takes 2-3 seconds. Subsequent are faster.

**Q: "No results from web?"**
A: Try more general queries like "python books" instead of specifics

**Q: "I see 'No module duckduckgo-search'"**
A: Run: `pip install -r requirements.txt`

**Q: "Web results have no price"**
A: That's expected. Web search provides snippets, not pricing.

**Q: "Can I use Google instead?"**
A: Yes! Replace duckduckgo with other search libraries. Same interface!

---

## NOW RUN:

```bash
python demo.py
```

And prepare to amaze your audience with a flexible AI system! 🚀
