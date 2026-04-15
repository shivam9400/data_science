"""
Visualization Helper
Generate a text-based flow diagram for presentations
"""

def show_workflow_diagram():
    """Display an ASCII flow diagram of the workflow"""
    
    diagram = """
╔════════════════════════════════════════════════════════════════════╗
║          SHOPPING ASSISTANT - LANGGRAPH WORKFLOW                 ║
╚════════════════════════════════════════════════════════════════════╝

                           📝 INPUT NODE
                              │
                    User Query: "cooking"
                              │
                              ▼
                           🔍 SEARCH NODE
                              │
                    Search catalog for matches
                    Found: 4 candidates
                              │
                              ▼
                           🎯 FILTER NODE
                              │
                    Apply preferences:
                    • Max price: $50
                    • Min rating: 4.3★
                    Filtered: 3 books
                              │
                              ▼
                         ✍️ SUMMARIZER NODE
                              │
                    Generate one-line pitches:
                    1) "Great for beginners"
                    2) "Perfect for advanced cooks"
                    3) "Healthy Mediterranean recipes"
                              │
                              ▼
                          📋 OUTPUT NODE
                              │
                    Display recommendations:
                    ├─ Title
                    ├─ Author
                    ├─ Price
                    ├─ Rating
                    └─ One-line pitch
                              │
                              ▼
                           ✅ DONE

╔════════════════════════════════════════════════════════════════════╗
║  Each node processes data, adds value, and passes to the next    ║
║  This makes the workflow MODULAR, TESTABLE, and EXTENSIBLE      ║
╚════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)


def show_state_flow():
    """Show how state flows through the system"""
    
    state_flow = """
╔════════════════════════════════════════════════════════════════════╗
║                    HOW STATE FLOWS                                ║
╚════════════════════════════════════════════════════════════════════╝

INPUT NODE:
    State {query: "cooking", original_query: "cooking"}
                    ↓ passes to
SEARCH NODE:
    State {candidates: [book1, book2, book3, book4]}
                    ↓ passes to
FILTER NODE:
    State {filtered_books: [book1, book2, book3]}
                    ↓ passes to
SUMMARIZER NODE:
    State {recommendations: [{title, pitch, price, rating}, ...]}
                    ↓ passes to
OUTPUT NODE:
    State {completed: true}

KEY CONCEPT: The same State object is modified by each node!
Each node adds or modifies data for the next node to use.
"""
    print(state_flow)


def show_quick_reference():
    """Quick reference for presentation"""
    
    reference = """
╔════════════════════════════════════════════════════════════════════╗
║            QUICK REFERENCE FOR PRESENTATIONS                      ║
╚════════════════════════════════════════════════════════════════════╝

1️⃣  INPUT NODE
    What: Receives user query
    Example: "Find me a cooking book"
    Output: Prepared query in state

2️⃣  SEARCH NODE
    What: Finds candidates in catalog
    Example: Searches 6 books, finds 4 matches
    Output: List of candidates

3️⃣  FILTER NODE
    What: Narrows based on preferences
    Example: Price < $50, Rating >= 4.3
    Output: Top 3 filtered books

4️⃣  SUMMARIZER NODE
    What: Creates compelling pitches
    Example: "Perfect for beginners"
    Output: Book + one-line summary

5️⃣  OUTPUT NODE
    What: Formats final recommendations
    Example: Title + Author + Price + Rating + Pitch
    Output: Beautiful formatted recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY ADVANTAGES:
✅ Each node is independent and testable
✅ Easy to explain each step
✅ Can visualize the flow
✅ Can add complexity gradually
✅ Real-world applicable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(reference)


if __name__ == "__main__":
    show_workflow_diagram()
    print("\n")
    show_state_flow()
    print("\n")
    show_quick_reference()
