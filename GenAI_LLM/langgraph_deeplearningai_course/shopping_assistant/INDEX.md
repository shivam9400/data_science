# 🛍️ SHOPPING ASSISTANT - PROJECT INDEX

Start here! This file tells you what to do with each file.

## 🚀 GET STARTED IN 3 STEPS

```
Step 1: INSTALL
   cd shopping_assistant
   pip install -r requirements.txt

Step 2: TEST
   python demo.py

Step 3: PRESENT
   Use the files below in order!
```

## 📚 YOUR PRESENTATION ROADMAP

**Read BEFORE Presenting:**
1. README.md (2 min) - Project overview
2. QUICK_START.md (5 min) - Pre-presentation checklist
3. PRESENTATION_GUIDE.md (10 min) - Full script with talking points

**Execute DURING Presentation:**
1. Show: README.md or visualize.py (explain the flow)
2. Run: `python visualize.py` (show diagrams)
3. Run: `python demo.py` (live demonstration)
4. Show: nodes/input_node.py (prove code is simple!)
5. Discuss: Real-world applications

**Reference AFTER Questions:**
- FILES_REFERENCE.md (what each file does)
- README.md → Extending section (how to add features)

## 📁 PROJECT FILES & WHAT THEY DO

### CORE WORKFLOW
- **workflow.py** - Main LangGraph orchestrator; Connects all nodes together; Contains ShoppingState definition. ⭐ This is the "brain"

### APPLICATION
- **demo.py** - Ready-to-run demonstrations; 3 different scenarios; Shows the workflow in action. ⭐ RUN THIS DURING YOUR PRESENTATION

### INDIVIDUAL NODES (each ~15-30 lines)
- **nodes/** - All individual workflow steps
  - input_node.py (Step 1: Receive query)
  - search_node.py (Step 2: Find candidates)
  - filter_node.py (Step 3: Apply preferences)
  - summarizer_node.py (Step 4: Create pitches)
  - output_node.py (Step 5: Display results)
- ⭐ Show these to prove each node is simple!

### DATA
- **data/books_catalog.py** - 6 sample books for demo
- **data/web_search.py** - Open-source web search integration
- ⭐ This is the "mock database"

### PRESENTATION MATERIALS
- **README.md** - Overview and quick start; Flow diagrams; Real-world examples; Questions to ask audience. ⭐ START WITH THIS
- **PRESENTATION_GUIDE.md** - Full script with talking points; Timing for each section; Audience questions; Live demo narration. ⭐ REFERENCE THIS DURING YOUR TALK
- **QUICK_START.md** - Pre-presentation checklist; Setup verification; Screen setup tips; Troubleshooting. ⭐ DO THIS BEFORE YOU PRESENT
- **FILES_REFERENCE.md** - What each file does; When to use it; File sizes and complexity. ⭐ SHARE THIS WITH INTERESTED PEOPLE
- **WEB_SEARCH_GUIDE.md** - Complete web search documentation
- **WEB_SEARCH_INTEGRATION_SUMMARY.md** - Summary of web search changes

### VISUALIZATIONS
- **visualize.py** - ASCII workflow diagrams; State flow visualization; Quick reference. ⭐ RUN THIS DURING PRESENTATION
- **generate_diagram.py** - Generate PNG diagrams (if graphviz installed)

### CONFIGURATION
- **requirements.txt** - Python packages (pip install)

## 🎯 THE 5-MINUTE QUICK VERSION

If you only have 5 minutes:

1. Show: README.md (1 min) - "Here's what we're building..."
2. Run: `python visualize.py` (1 min) - "Here's how it works..."
3. Run: `python demo.py` (2 min) - "Watch it in action..."
4. Show: nodes/input_node.py (1 min) - "Each part is really simple!"

**Done!** They'll understand the concept.

## 🎯 THE 20-MINUTE FULL VERSION

If you have 20 minutes:

1. **Introduction** (2 min)
   - Problem: Too many choices for users
   - Solution: AI-guided workflow

2. **Show Architecture** (3 min)
   - README.md
   - visualize.py output
   - Explain each node

3. **Code Walk-through** (3 min)
   - Show nodes/ folder
   - Prove each is small (15-30 lines)
   - Show workflow.py

4. **Live Demo** (5 min)
   - Run: `python demo.py`
   - Narrate each step
   - Maybe run again with different parameters

5. **Discussion** (5 min)
   - Real-world applications
   - How to extend
   - Q&A

## ✨ PRESENTATION TIPS

### DO THIS:
- Start with the problem (why this matters)
- Use the diagrams (visualize.py)
- Run the demo live (it works!)
- Show code is simple (nodes/)
- Relate to their business
- Ask for questions
- Show enthusiasm!

### DON'T DO THIS:
- Explain every line of code
- Use too much jargon
- Read from notes verbatim
- Spend too long on technicalities
- Forget to show why it matters
- Rush through the demo

## 🔧 QUICK REFERENCE

Run the demo:
```bash
python demo.py
```

Show diagrams:
```bash
python visualize.py
```

See file details:
```bash
python FILES_REFERENCE.py
```

Show presentation guide:
```bash
python PRESENTATION_GUIDE.py
```

Show checklist:
```bash
python QUICK_START.py
```

Generate PNG diagram (needs graphviz):
```bash
python generate_diagram.py
```

## ❓ COMMON QUESTIONS

**Q: "How long does it take to present?"**
A: 15-20 minutes is ideal. Can be 5 minutes if rushed or 30+ if interactive.

**Q: "Does it need internet?"**
A: No! It works completely offline with mock data (catalog mode).

**Q: "Can I modify it?"**
A: Yes! See README.md → Extending the Project

**Q: "What if I want real data?"**
A: Replace books_catalog.py with database queries

**Q: "What if I want to use a real AI model?"**
A: Replace the pitch logic in summarizer_node.py with an LLM call

**Q: "Can I show this on a big screen?"**
A: Yes! Terminal output is clear, demo runs smoothly

**Q: "How do I explain this to my boss?"**
A: Use PRESENTATION_GUIDE.md → Real World Applications section

## 🎓 LEARNING OUTCOMES FOR YOUR AUDIENCE

After your presentation, people will understand:

1. **What LangGraph is** → Framework for building AI workflows
2. **Why nodes are useful** → Each does one thing, easy to test, easy to modify
3. **How state flows through a system** → Same object gets enhanced at each step
4. **Real-world applications** → This pattern appears everywhere in business
5. **That AI workflows don't have to be scary** → Each piece is simple, together they're powerful

---

## NOW YOU'RE READY!

**Next step:** Read QUICK_START.md, then present!

**Good luck!** 🚀
