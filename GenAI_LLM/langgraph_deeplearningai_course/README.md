# 🛍️ Shopping Assistant - LangGraph Demo

A simple, non-technical LangGraph project perfect for demonstrating AI workflows to a general audience.

**NEW:** Now includes open-source web search capability! 🌐

## Project Overview

This is a **Shopping Assistant** that helps users find the best books based on their preferences. The project demonstrates core LangGraph concepts in an easy-to-understand way.

### The Flow

```
1. INPUT NODE
   └─> Takes user query (e.g., "Find me a cooking book")

2. SEARCH NODE
   └─> Searches: Local catalog, Web, or Both 🌐
   └─> Returns matching candidates

3. FILTER NODE
   └─> Applies price and rating filters
   └─> Narrows down to top 2-3 books

4. SUMMARIZER NODE
   └─> Generates compelling one-line pitches
   └─> "Perfect for beginners wanting to learn fast"
   └─> "Ideal for advanced cooks exploring global cuisines"

5. OUTPUT NODE
   └─> Presents final recommendations with:
       - Title, Author, Price, Rating
       - Personalized pitch
       - Web links (for web results)
```

## Project Structure

```
shopping_assistant/
├── data/
│   └── books_catalog.py          # Mock dataset (6 sample books)
├── nodes/
│   ├── __init__.py
│   ├── input_node.py             # Step 1: Receive user query
│   ├── search_node.py            # Step 2: Search catalog
│   ├── filter_node.py            # Step 3: Apply preferences
│   ├── summarizer_node.py        # Step 4: Generate pitches
│   └── output_node.py            # Step 5: Display results
├── workflow.py                    # Main LangGraph workflow
├── demo.py                        # Run 3 demo scenarios
├── requirements.txt              # Dependencies
└── README.md                      # This file
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langgraph` - The workflow orchestration framework
- `langchain` - LLM utilities
- `duckduckgo-search` - Open-source web search (no API key needed!) 🌐

### 2. Run the Demo

```bash
python demo.py
```

### 3. Try It Yourself

```python
from workflow import run_shopping_assistant

# Example 1: Search local catalog
run_shopping_assistant(
    query="cooking",
    search_source="catalog"
)

# Example 2: Search the web
run_shopping_assistant(
    query="python programming",
    search_source="web"
)

# Example 3: Hybrid search (catalog + web)
run_shopping_assistant(
    query="cooking",
    search_source="both"
)
```

## Demo Scenarios

The `demo.py` file includes 4 pre-built examples:

- **Demo 1**: Search local catalog
- **Demo 2**: Search the web 🌐
- **Demo 3**: Hybrid search (catalog + web) 🌐
- **Demo 4**: Budget-friendly options from catalog

## Key Concepts Demonstrated

✅ **Node-based Architecture** - Each step is a separate, testable node
✅ **State Management** - Data flows smoothly through the graph
✅ **Linear Workflow** - Simple sequence: input → search → filter → summarize → output
✅ **Modularity** - Easy to swap nodes or add new ones
✅ **Non-Technical** - Uses everyday language and emojis for clarity
✅ **Open-Source Web Search** - No API keys needed! Search the web with DuckDuckGo 🌐


## Web Search Feature 🌐

The Shopping Assistant now supports **open-source web search** using DuckDuckGo!

### Three Search Modes:

**1. Catalog Only** (Default)
- Search the local dataset (6 books)
- Fast and deterministic
- Good for demos

**2. Web Search** 
- Search the entire internet
- Real-time results
- No pricing/rating data

**3. Hybrid** (Catalog + Web)
- Best of both worlds
- Catalog results first, then web results
- Comprehensive results

### Example:

```python
# Web search for Python books
run_shopping_assistant(
    query="python programming",
    search_source="web"
)

# Hybrid: catalog + web together
run_shopping_assistant(
    query="cooking",
    search_source="both",
    max_price=50.0,
    min_rating=4.3
)
```

See [WEB_SEARCH_GUIDE.py](WEB_SEARCH_GUIDE.py) for detailed documentation!

## Why This Project is Great for Presentations

1. **Relatable Domain** - Everyone understands shopping and product recommendations
2. **Visual Flow** - Clear step-by-step progression
3. **Easy to Explain** - Each node does one simple thing
4. **Mockable Data** - No external APIs needed
5. **Interactive** - Can modify queries and filters on the fly
6. **Extensible** - Easy to add features (reviews, comparisons, etc.)

## Extending the Project

Want to make it more advanced?

- Add **conditional routing** (different flows for budget vs premium searches)
- Add **human feedback** nodes
- Add **real LLM integration** for dynamic pitch generation
- Add **parallel nodes** for faster processing
- Add **error handling** and retry logic
- Swap **web search provider** (Google, Bing, etc.)
- Add **price extraction** from web results
- Add **review scraping** from search results
- Add **caching** for repeated searches

## Notes for Your Presentation

📍 **Start simple**: Show the basic flow first
📍 **Walk through code**: Each node is only 10-15 lines
📍 **Run demos live**: Show how filtering changes recommendations
📍 **Ask the audience**: "What if we added X feature?"
📍 **Show visualization**: Use `graph.draw_png()` to create a visual diagram

## Questions to Ask Your Audience

- "What happens if no books match the search?"
- "Where would we add AI to generate better pitches?"
- "How could we handle multiple types of products?"
- "Where could this pattern apply in your industry?"

---

**Happy presenting! 🎉**
