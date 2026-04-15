"""
Step 2: Search Node
Searches through catalog, web, or both for matching books
"""

from data.books_catalog import BOOKS_CATALOG
from data.web_search import search_web_for_books

def search_node(state):
    """
    Search for books matching user query
    Can search: catalog, web, or both
    """
    query = state.get("query", "").lower()
    search_source = state.get("search_source", "catalog")  # catalog, web, or both
    
    print(f"📚 SEARCH: Looking for books matching '{query}'")
    print(f"   Source: {search_source}")
    
    candidates = []
    
    # Search catalog if requested
    if search_source in ["catalog", "both"]:
        catalog_results = []
        for book in BOOKS_CATALOG:
            if (query in book["title"].lower() or 
                query in book["author"].lower() or 
                query in book["description"].lower()):
                catalog_results.append(book)
        
        if not catalog_results:
            catalog_results = BOOKS_CATALOG[:2]
        
        candidates.extend(catalog_results)
        print(f"   📖 Catalog: {len(catalog_results)} books")
    
    # Search web if requested
    if search_source in ["web", "both"]:
        web_results = search_web_for_books(query, max_results=3)
        candidates.extend(web_results)
        print(f"   🌐 Web: {len(web_results)} results")
    
    if not candidates:
        candidates = BOOKS_CATALOG[:4]
        print(f"   No matches found. Showing popular books...")
    
    print(f"   Total candidates: {len(candidates)}")
    state["candidates"] = candidates
    state["search_source"] = search_source
    return state
