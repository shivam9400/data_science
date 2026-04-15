"""
Generate visual diagram of the LangGraph workflow
Requires: pip install pygraphviz (optional, for PNG export)
"""

from workflow import build_workflow


def draw_workflow():
    """
    Draw the workflow graph and save as image
    
    Note: This requires graphviz to be installed on your system
    For presentations, you can take a screenshot of this
    """
    try:
        graph = build_workflow()
        
        # ASCII art visualization (always works)
        print("\n" + "="*60)
        print(" LangGraph Shopping Assistant - Structure")
        print("="*60 + "\n")
        
        print(graph.get_graph().draw_ascii())
        
        print("\n" + "="*60)
        print(" Graph structure displayed above!")
        print("="*60)
        
        # Try to generate PNG (requires graphviz)
        try:
            png_data = graph.get_graph().draw_mermaid_png()
            with open("shopping_assistant_workflow.png", "wb") as f:
                f.write(png_data)
            print("\n✅ PNG image saved: shopping_assistant_workflow.png")
        except Exception as e:
            print(f"\n⚠️  Could not generate PNG (graphviz not installed): {e}")
            print("   But ASCII diagram above works great for presentations!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure langgraph is properly installed")


if __name__ == "__main__":
    draw_workflow()
