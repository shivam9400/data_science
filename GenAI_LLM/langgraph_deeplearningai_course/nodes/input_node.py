"""
Step 1: Input Node
Takes user query and prepares it for processing
"""

def input_node(state):
    """
    Entry point - receives user query
    """
    # For demo purposes, user query is passed in state
    query = state.get("query", "")
    print(f"🔍 INPUT: User query: '{query}'")
    
    state["original_query"] = query
    return state
