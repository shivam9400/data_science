"""
Demo Script: Shopping Assistant
Simple examples to show the workflow in action
Includes: Catalog search, Web search, and Combined search
"""

from workflow import run_shopping_assistant

def demo_1():
    """Demo 1: Search local catalog"""
    print("\n" + "🎬 " + "DEMO 1: Search local catalog")
    print("-" * 60)
    run_shopping_assistant(
        query="cooking",
        max_price=50.0,
        min_rating=4.3,
        search_source="catalog"
    )


def demo_2():
    """Demo 2: Search the web"""
    print("\n" + "🎬 " + "DEMO 2: Search the web")
    print("-" * 60)
    run_shopping_assistant(
        query="python programming",
        max_price=100.0,
        min_rating=4.0,
        search_source="web"
    )


def demo_3():
    """Demo 3: Search both catalog and web"""
    print("\n" + "🎬 " + "DEMO 3: Combined search (catalog + web)")
    print("-" * 60)
    run_shopping_assistant(
        query="beginner",
        max_price=50.0,
        min_rating=4.3,
        search_source="both"
    )


def demo_4():
    """Demo 4: Budget-friendly from catalog"""
    print("\n" + "🎬 " + "DEMO 4: Budget-friendly books (catalog only)")
    print("-" * 60)
    run_shopping_assistant(
        query="book",
        max_price=25.0,
        min_rating=4.4,
        search_source="catalog"
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SHOPPING ASSISTANT - LangGraph Demo")
    print(" (With Local Catalog & Web Search)")
    print("="*60)
    
    # Run all demos
    demo_1()
    demo_2()
    demo_3()
    demo_4()
    
    print("\n" + "="*60)
    print(" All demos completed! ✅")
    print("="*60)
