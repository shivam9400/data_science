# WEB SEARCH FEATURE - DUCKDUCKGO INTEGRATION

Complete guide to the open-source web search integration

---

## 🌐 WHAT IS WEB SEARCH?

The Shopping Assistant can now search the web for books, not just the local catalog. This gives users access to:

- Real-time book information
- Current pricing and availability
- Latest books and recommendations
- Unlimited product variety

**All with ZERO API keys needed!**

---

## 🔧 HOW IT WORKS

### Technology
- **Search Engine:** DuckDuckGo API
- **Authentication:** Open-source friendly, no auth required
- **Python Library:** duckduckgo-search (lightweight wrapper)
- **Installation:** Already in requirements.txt

### Implementation Files
- `data/web_search.py` - Web search functions
- `search_node.py` - Modified to support web search
- `filter_node.py` - Handles web results
- `summarizer_node.py` - Creates pitches from snippets
- `output_node.py` - Displays web links

---

## 🎯 SEARCH SOURCES

### Three Modes

**1. CATALOG ONLY (Default)**
```python
run_shopping_assistant(
    query="cooking",
    search_source="catalog"
)
```
- Search only the local dataset (6 books)
- Fast, deterministic
- Good for demos

**2. WEB ONLY**
```python
run_shopping_assistant(
    query="python programming",
    search_source="web"
)
```
- Search the entire web
- Live results
- Unlimited variety

**3. BOTH (Hybrid)**
```python
run_shopping_assistant(
    query="cooking",
    search_source="both"
)
```
- Search catalog first
- Combine with web results
- Best of both worlds

---

## 💻 CODE EXAMPLES

### Example 1: Simple web search
```python
from workflow import run_shopping_assistant

result = run_shopping_assistant(
    query="machine learning books",
    search_source="web"
)
```

### Example 2: Hybrid search with budget
```python
result = run_shopping_assistant(
    query="cooking",
    max_price=50.0,
    search_source="both"
)
```

### Example 3: Direct web search function
```python
from data.web_search import search_web_for_books

books = search_web_for_books("data science", max_results=5)
for book in books:
    print(book["title"])
    print(book["snippet"])
```

---

## 📊 RESULTS COMPARISON

### CATALOG SEARCH
| Pros | Cons |
|------|------|
| Fast | Limited |
| Consistent | Fixed data |
| Reliable | Only 6 books |

### WEB SEARCH
| Pros | Cons |
|------|------|
| Unlimited | Slower |
| Live data | Varied |
| Current | No pricing |

### HYBRID SEARCH
| Pros | Cons |
|------|------|
| Best variety | Mixed types |
| Fast + Live | Complex |
| Backups | More to flag |

---

## 📊 DATA DIFFERENCES

### CATALOG RESULTS have:
- ✅ Title
- ✅ Author
- ✅ Price (USD)
- ✅ Rating (0-5 stars)
- ✅ Description

### WEB RESULTS have:
- ✅ Title
- ✅ Snippet (from search result)
- ✅ Link (to original source)
- ❌ Price (not available)
- ❌ Rating (not available)

---

## 🔌 INSTALLATION

Web search requires one additional package:

```bash
pip install duckduckgo-search
```

This is already in requirements.txt, so:

```bash
pip install -r requirements.txt
```

---

## 🚀 RUNNING WITH WEB SEARCH

Demo with all three search modes:

```bash
python demo.py
```

Individual demos:
- Demo 1: Catalog-only
- Demo 2: Web-only
- Demo 3: Hybrid (catalog + web)
- Demo 4: Budget filtering

---

## 📝 DEMO OUTPUT

You'll see different sources indicated:

### 📚 Catalog Results:
- Have: Title, Author, Price, Rating
- Show: Predefined pitches

### 🌐 Web Results:
- Have: Title, Snippet, Link
- Show: Snippet as pitch

### Mixed Results:
- Catalog items sorted by rating first
- Web results added after

---

## 🔍 SEARCH TIPS

### Good queries
- ✓ "python programming"
- ✓ "machine learning"
- ✓ "cooking for beginners"
- ✓ "data science"

### Better queries (more specific)
- ✓ "introduction to python books"
- ✓ "best machine learning books 2024"
- ✓ "beginner cooking books"

### Advanced searches
- ✓ Use market names: "cookbooks for vegetarians"
- ✓ Include author: "books by tim ferriss"
- ✓ Include year: "data science books 2024"

---

## ⚙️ CUSTOMIZATION

### Want to modify web search behavior?

**1. Edit data/web_search.py to:**
   - Change results format
   - Filter by source domain
   - Extract pricing data
   - Add custom ranking

**2. Modify search_node.py to:**
   - Change number of results
   - Adjust search query
   - Mix catalog and web differently

**3. Update filter_node.py to:**
   - Handle missing data differently
   - Create separate pipelines
   - Add web-specific filters

---

## 🎓 FOR YOUR PRESENTATION

### Talking Points

1. **"We can search local data OR the web"**
   - Demonstrate switching between sources

2. **"Web search gives infinite data"**
   - Show a web search result

3. **"Same workflow, different data sources"**
   - Explain modularity

4. **"No API keys needed"**
   - Emphasize ease of use

5. **"Easy to swap search implementations"**
   - Show it's composable

---

## 🔗 RELATED DOCUMENTATION

- **WEB_SEARCH_INTEGRATION_SUMMARY.md** - What changed
- **QUICK_START_WEB_SEARCH.md** - Quick setup
- **README.md** - Main project documentation

---

## 📞 SUPPORT

### Common Issues

**Q: Web search is slow**
A: Normal! First request takes 2-3 seconds. Cached results are faster.

**Q: No results returned**
A: Try more general query. "python books" works better than very specific titles.

**Q: Module not found: duckduckgo_search**
A: Run `pip install -r requirements.txt` again

**Q: Web results show no price**
A: Expected behavior. Web search provides snippets, not Commerce data.

**Q: Can I use Google instead?**
A: Yes! Replace with google-search or similar. Interface stays the same.

---

**Ready to search the web?** 🚀

Run: `python demo.py` to see it in action!
