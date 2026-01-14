''' defines sequence of operations '''

from langgraph.graph import StateGraph, START, END
from state import FMCGState
from nodes.search import search_node
from nodes.crawl import crawl_node
from nodes.validate import validate_scrape_node, route_after_validation
from nodes.classify import classify_node

workflow = StateGraph(FMCGState)

workflow.add_node("search_scout", search_node)
workflow.add_node("web_reader", crawl_node)
workflow.add_node("validater", validate_scrape_node)
workflow.add_node("ai_classifier", classify_node)

workflow.set_entry_point("search_scout")
workflow.add_edge("search_scout", "web_reader")
workflow.add_edge("web_reader", "validater")
# workflow.add_edge("web_reader", "ai_classifier")
# workflow.add_edge("ai_classifier", END)

# Validator logic
workflow.add_conditional_edges(
    "validater",
    route_after_validation,
    {
        "retry": "web_reader",      # Loops back! Ensure scraper increment index
        "process": "ai_classifier", # Moves forward
        "end": END                  # Stops
    }
)

# THE REACT LOOP: 
def should_continue(state):
    # If index increased, we go back to crawl a new URL
    if state["current_url_index"] < len(state["found_urls"]) and not state["final_output"]:
        return "retry"
    return "end"

workflow.add_conditional_edges(
    "ai_classifier",
    should_continue,
    {
        "retry": "web_reader", # Loop back to crawl with the next index
        "end": END
    }
)

# compile the graph
app = workflow.compile()