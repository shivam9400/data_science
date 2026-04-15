# FILES REFERENCE

Complete reference of all files in the Shopping Assistant project and what they do.

## 🗂️ PROJECT STRUCTURE

```
shopping_assistant/
├── data/
│   └── books_catalog.py              ← 6 sample books (mock data)
├── nodes/
│   ├── __init__.py                   ← Package marker
│   ├── input_node.py                 ← Node 1: Receive query
│   ├── search_node.py                ← Node 2: Search catalog
│   ├── filter_node.py                ← Node 3: Apply filters
│   ├── summarizer_node.py            ← Node 4: Generate pitches
│   └── output_node.py                ← Node 5: Display results
├── workflow.py                        ← Main LangGraph workflow
├── demo.py                            ← Run 3 demo scenarios
├── visualize.py                       ← Show ASCII diagrams
├── generate_diagram.py                ← Generate graph images
├── PRESENTATION_GUIDE.md              ← Talking points & script
├── QUICK_START.md                     ← Checklist before presenting
├── requirements.txt                   ← Python dependencies
└── README.md                          ← Project overview
```

## 📄 FILES EXPLAINED

### 1. data/books_catalog.py

**Contains:** BOOKS_CATALOG list with 6 sample books

**Each book has:**
- id, title, author, price, rating
- category, description

**Used by:** search_node.py

**Purpose:** Mock data source (simulates database)

---

### 2. nodes/input_node.py (~15 lines)

**Function:** `input_node(state)`

**Takes:** User query from state

**Returns:** State with original_query stored

**Purpose:** Entry point, prepare query for processing

---

### 3. nodes/search_node.py (~20 lines)

**Function:** `search_node(state)`

**Takes:** Query from state

**Returns:** candidates list in state

**Does:** Searches catalog for matching books

**Purpose:** Find initial candidates

---

### 4. nodes/filter_node.py (~25 lines)

**Function:** `filter_node(state)`

**Takes:** candidates + max_price + min_rating

**Returns:** filtered_books in state

**Does:** Applies price and rating filters, sorts by rating

**Purpose:** Narrow down to best matches

---

### 5. nodes/summarizer_node.py (~30 lines)

**Function:** `summarizer_node(state)`

**Takes:** filtered_books

**Returns:** recommendations with pitches

**Does:** Generates one-line pitch for each book

**Purpose:** Make recommendations compelling

---

### 6. nodes/output_node.py (~20 lines)

**Function:** `output_node(state)`

**Takes:** recommendations

**Returns:** Same state with completed=true

**Does:** Formats and prints final results

**Purpose:** Display recommendations to user

---

### 7. workflow.py (~80 lines)

**Defines:** ShoppingState class (TypedDict with all data fields)

**Functions:**
- `build_workflow()` - creates LangGraph
- `run_shopping_assistant()` - executes the workflow

**Connections:** START → input → search → filter → summarizer → output → END

**Purpose:** Main orchestrator that ties all nodes together

---

### 8. demo.py (~40 lines)

**Functions:**
- `demo_1()` - General cooking search
- `demo_2()` - Budget-friendly books
- `demo_3()` - Premium high-rated books

**Calls:** `run_shopping_assistant()` with different parameters

**Purpose:** Show the workflow in action (run this for live demo!)

---

### 9. visualize.py (~80 lines)

**Functions:**
- `show_workflow_diagram()` - ASCII flow diagram
- `show_state_flow()` - How state is modified
- `show_quick_reference()` - Quick facts

**Purpose:** Text-based visualizations for presentations

---

### 10. generate_diagram.py (~30 lines)

**Function:** `draw_workflow()` - Creates ASCII and PNG diagrams

**Generates:** shopping_assistant_workflow.png (if graphviz installed)

**Purpose:** Generate visual diagrams for slides

---

### 11. PRESENTATION_GUIDE.md (~200 lines)

**Contains:**
- Full presentation script with talking points
- Sections: Intro, Concept, Architecture, Demo narration
- Real-world examples
- Audience questions

**Purpose:** Your reference doc during presentation

---

### 12. QUICK_START.md (~120 lines)

**Includes:**
- Checklist: What to do before presenting
- During: Step-by-step presentation flow
- Troubleshooting: Common issues and fixes

**Purpose:** Pre-presentation checklist

---

### 13. README.md (~150 lines)

**Contains:**
- Project overview
- Folder structure
- Getting started guide
- Flow diagram
- Key concepts
- Extension ideas

**Purpose:** Project documentation & reference

---

### 14. requirements.txt

**Contains:**
- langgraph==0.0.65
- langchain==0.1.13

**Purpose:** Python dependencies (install with: `pip install -r requirements.txt`)

---

## 🔄 DATA FLOW

```
User Query
    ↓ (input_node)
Query in state
    ↓ (search_node)
Candidates from catalog (4-6 books)
    ↓ (filter_node)
Filtered candidates (top 2-3 books)
    ↓ (summarizer_node)
Recommendations with one-line pitches
    ↓ (output_node)
Beautiful formatted output to user
```

## 🎯 WHEN TO USE EACH FILE

### BEFORE PRESENTATION
```
□ Run: python QUICK_START.py (checklist)
□ Read: PRESENTATION_GUIDE.md (memorize talking points)
□ Test: python demo.py (make sure it works)
```

### DURING PRESENTATION
```
□ Show: README.md (overview)
□ Run: python visualize.py (show diagrams)
□ Run: python demo.py (live demo)
□ MAYBE: Show nodes/input_node.py, workflow.py (show code is simple)
```

### IF SOMEONE ASKS
```
"How do I run it?" → Share: QUICK_START.md
"What's the flow?" → Show: visualize.py output
"How do I extend it?" → Show: README.md → Extending section
"Can I use this for X?" → Use: PRESENTATION_GUIDE.md → Real-world examples
```

## 📊 FILE SIZES & COMPLEXITY

### Small & Simple (one node each)
- input_node.py (~15 lines)
- search_node.py (~20 lines)
- filter_node.py (~25 lines)
- output_node.py (~20 lines)
- summarizer_node.py (~30 lines)

### Medium (orchestration)
- workflow.py (~80 lines)

### Demo & Help
- demo.py (~40 lines)
- visualize.py (~80 lines)
- PRESENTATION_GUIDE.md (~200 lines)

### Reference
- README.md (~150 lines)
- data/books_catalog.py (~30 lines listing data)

## 💡 WHAT MAKES THIS PRESENTATION-READY

✅ Each node is short & understandable  
✅ Multiple demo scripts (easy to run)  
✅ Clear documentation (non-technical)  
✅ Visual aids included (diagrams, ASCII art)  
✅ Talking points prepared (presentation guide)  
✅ Real-world examples (in README)  
✅ Extensibility explained (easy to modify)  
✅ No external dependencies (just langgraph/langchain)  

---
