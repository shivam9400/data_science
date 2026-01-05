''' defines sequence of operations '''

from langgraph.graph import StateGraph, END
from state import FMCGState
from nodes.search import search_node
from nodes.crawl import crawl_node
from nodes.classify import classify_node

workflow = StateGraph(FMCGState)

workflow.add_node("search_scout", search_node)
workflow.add_node("web_reader", crawl_node)
workflow.add_node("ai_classifier", classify_node)

workflow.set_entry_point("search_scout")
workflow.add_edge("search_scout", "web_reader")
workflow.add_edge("web_reader", "ai_classifier")
workflow.add_edge("ai_classifier", END)

app = workflow.compile()