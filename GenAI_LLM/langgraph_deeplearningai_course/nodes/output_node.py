"""
Step 5: Output Node
Formats and presents final recommendations
Handles both catalog and web results
"""

def output_node(state):
    """
    Format and display final recommendations
    Includes pricing info for catalog items and links for web results
    """
    recommendations = state.get("recommendations", [])
    search_source = state.get("search_source", "catalog")
    
    print("\n" + "="*60)
    print("📋 FINAL RECOMMENDATIONS")
    print("="*60)
    
    if not recommendations:
        print("❌ No recommendations found. Try a different search!")
    else:
        for idx, rec in enumerate(recommendations, 1):
            print(f"\n{idx}. {rec['title']}")
            print(f"   Author: {rec['author']}")
            
            # Handle different sources
            if rec.get("source") == "web":
                print(f"   Price: Check online")
                print(f"   Rating: From web search")
                if rec.get("link"):
                    print(f"   Link: {rec['link']}")
            else:
                price = rec['price']
                if isinstance(price, (int, float)):
                    print(f"   Price: ${price:.2f}")
                else:
                    print(f"   Price: {price}")
                    
                rating = rec['rating']
                if isinstance(rating, (int, float)):
                    print(f"   Rating: {'⭐' * int(rating)} ({rating}/5)")
                else:
                    print(f"   Rating: {rating}")
            
            print(f"   Summary: {rec['pitch']}")
    
    print("\n" + "="*60)
    
    # Show search source in summary
    if search_source == "both":
        print("📝 Results include: Local catalog + Web search")
    elif search_source == "web":
        print("📝 Results from: Web search")
    else:
        print("📝 Results from: Local catalog")
    
    print("="*60)
    state["completed"] = True
    return state
