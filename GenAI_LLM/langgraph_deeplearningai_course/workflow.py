"""
Shopping Assistant Workflow
LangGraph implementation that orchestrates the entire flow
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Import all nodes
from nodes.input_node import input_node
from nodes.search_node import search_node
from nodes.filter_node import filter_node
from nodes.summarizer_node import summarizer_node
from nodes.output_node import output_node


# Define the state structure
class ShoppingState(TypedDict):
    """
    State that flows through the workflow
    """
    query: str                          # User's search query
    original_query: str                 # Original unmodified query
    search_source: str                  # "catalog", "web", or "both"
    max_price: float                    # Maximum price threshold
    min_rating: float                   # Minimum rating threshold
    candidates: list                    # Initial search results
    filtered_books: list                # After filtering by price/rating
    recommendations: list               # Final recommendations with pitches
    completed: bool                     # Workflow completion flag


def build_workflow():
    """
    Build the LangGraph workflow
    """
    # Create the graph
    workflow = StateGraph(ShoppingState)
    
    # Add all nodes (step by step)
    workflow.add_node("input", input_node)
    workflow.add_node("search", search_node)
    workflow.add_node("filter", filter_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("output", output_node)
    
    # Define edges (connections between nodes)
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "search")
    workflow.add_edge("search", "filter")
    workflow.add_edge("filter", "summarizer")
    workflow.add_edge("summarizer", "output")
    workflow.add_edge("output", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    return graph


def run_shopping_assistant(query, max_price=50.0, min_rating=4.3, search_source="catalog"):
    """
    Execute the shopping assistant workflow
    
    Args:
        query: User's search query (e.g., "cooking")
        max_price: Maximum price filter
        min_rating: Minimum rating filter
        search_source: Where to search - "catalog", "web", or "both"
    
    Returns:
        Final state with recommendations
    """
    # Build the workflow
    graph = build_workflow()
    
    # Initialize state
    initial_state = {
        "query": query,
        "original_query": query,
        "search_source": search_source,
        "max_price": max_price,
        "min_rating": min_rating,
        "candidates": [],
        "filtered_books": [],
        "recommendations": [],
        "completed": False
    }
    
    # Execute the workflow
    print("\n" + "🚀 " + "Starting Shopping Assistant Workflow...")
    print("=" * 60)
    
    final_state = graph.invoke(initial_state)
    
    return final_state
