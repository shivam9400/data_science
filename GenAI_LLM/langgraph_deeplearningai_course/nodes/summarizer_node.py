"""
Step 4: Summarizer Agent Node
Generates a one-line pitch for each filtered book
Handles both catalog and web results
"""

def summarizer_node(state):
    """
    Generates compelling one-line pitches for each book
    Uses pre-defined pitches for catalog items, snippets for web results
    """
    filtered_books = state.get("filtered_books", [])
    
    print(f"✍️  SUMMARIZER: Creating pitches for {len(filtered_books)} books")
    
    # Pre-defined pitches for catalog items
    pitches = {
        1: "Perfect for beginners who want to master the basics with confidence.",
        2: "Ideal for advanced cooks looking to explore professional techniques.",
        3: "Great for busy people who want delicious meals in 30 minutes.",
        4: "Discover healthy Mediterranean recipes that are both simple and elegant.",
        5: "Budget-friendly recipes perfect for feeding the whole family well.",
        6: "Learn to bake professional-quality desserts with step-by-step guidance."
    }
    
    summaries = []
    for book in filtered_books:
        # Use web snippet if available, otherwise use predefined pitch
        if book.get("source") == "web":
            pitch = book.get("snippet", "Interesting book found online")[:100] + "..."
        else:
            pitch = pitches.get(book.get("id"), "Great book for learning!")
        
        summary = {
            "title": book.get("title", "Unknown Title"),
            "author": book.get("author", "Unknown"),
            "price": book.get("price", "Check online"),
            "rating": book.get("rating", "Not rated"),
            "pitch": pitch,
            "link": book.get("link", ""),
            "source": book.get("source", "catalog")
        }
        summaries.append(summary)
        print(f"   📖 {book.get('title', 'Unknown')}")
        print(f"      💬 {pitch[:60]}...")
    
    state["recommendations"] = summaries
    return state
