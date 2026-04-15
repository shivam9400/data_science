"""
Step 3: Filter Node
Applies user preferences (price range, minimum rating)
Handles both catalog and web results
"""

def filter_node(state):
    """
    Filter candidates based on price and rating preferences
    Web results are kept if they pass basic checks
    """
    candidates = state.get("candidates", [])
    
    # User preferences (can be made interactive)
    max_price = state.get("max_price", 50.0)
    min_rating = state.get("min_rating", 4.3)
    
    print(f"🎯 FILTER: Applying preferences (max price: ${max_price}, min rating: {min_rating})")
    
    filtered = []
    for book in candidates:
        # Handle web results (no price/rating)
        if book.get("source") == "web":
            filtered.append(book)
        # Handle catalog results (with price/rating)
        else:
            if book.get("price", max_price) <= max_price and book.get("rating", min_rating) >= min_rating:
                filtered.append(book)
    
    # Sort: catalog items first (rated), then web results
    catalog_items = [b for b in filtered if b.get("source") != "web"]
    web_items = [b for b in filtered if b.get("source") == "web"]
    
    # Sort catalog by rating
    catalog_items = sorted(catalog_items, key=lambda x: x.get("rating", 0), reverse=True)
    
    # Combine and keep top items
    filtered = catalog_items + web_items
    filtered = filtered[:3]
    
    print(f"   Filtered down to {len(filtered)} books")
    state["filtered_books"] = filtered
    return state
