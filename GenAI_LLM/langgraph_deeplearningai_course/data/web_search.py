"""
Web Search Module
Uses DuckDuckGo (open-source, no API key) to search the web for books
"""

from typing import List, Dict


def search_web_for_books(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web for books using DuckDuckGo
    
    Args:
        query: Search query (e.g., "cooking books")
        max_results: Maximum number of results to return
    
    Returns:
        List of search results with title, link, and snippet
    """
    try:
        from ddgs import DDGS
    except ImportError:
        print("⚠️  ddgs not installed. Install with: pip install ddgs")
        return []
    
    try:
        ddg = DDGS()
        results = ddg.text(f"{query} book", max_results=max_results)
        
        # Convert to our format
        formatted_results = []
        for idx, result in enumerate(results, 1):
            formatted_results.append({
                "id": 1000 + idx,  # IDs starting from 1000 to distinguish from catalog
                "title": result.get("title", "Unknown Title"),
                "link": result.get("href", ""),
                "snippet": result.get("body", ""),
                "source": "web",
                "price": None,  # Web results don't have prices
                "rating": None,  # Web results don't have ratings
                "description": result.get("body", ""),
                "author": "Unknown"
            })
        
        return formatted_results
    
    except Exception as e:
        print(f"⚠️  Web search error: {e}")
        return []


def search_web_for_book_info(title: str, author: str = "") -> Dict:
    """
    Search the web for specific book information
    
    Args:
        title: Book title
        author: Book author (optional)
    
    Returns:
        Dictionary with book information from web results
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return {}
    
    try:
        ddg = DDGS()
        query = f"{title} {author}".strip()
        results = ddg.text(query, max_results=1)
        
        if results:
            result = results[0]
            return {
                "title": result.get("title", title),
                "link": result.get("href", ""),
                "snippet": result.get("body", ""),
                "source": "web"
            }
        
        return {}
    
    except Exception as e:
        print(f"⚠️  Web search error: {e}")
        return {}
