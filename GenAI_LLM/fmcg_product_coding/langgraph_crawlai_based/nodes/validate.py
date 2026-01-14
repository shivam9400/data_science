''' Inspects the scraped content and flags it as valid or invalid. '''

def validate_scrape_node(state):
    """
    NODE
    """
    content = state.get("scraped_content", "").lower()
    
    # Define indicators for a failed scrape (Amazon bot check, etc.)
    bot_markers = ["robot check", "captcha", "continue shopping", "access denied"]
    
    is_blocked = any(marker in content for marker in bot_markers)
    is_empty = len(content.strip()) < 200  # Usually a fail if < 200 chars
    
    # We update the state with the result
    return {
        "is_valid": not (is_blocked or is_empty)
    }

def route_after_validation(state):
    """
    ROUTER: Pure logic function for conditional edges.
    """
    if state.get("is_valid"):
        return "process"  # Proceed to classifier.py
    
    # If invalid, check if we have more URLs left in state['found_urls']
    current_idx = state.get("current_url_index", 0)
    if current_idx + 1 < len(state.get("found_urls", [])):
        return "retry"    # Loop back to scraper.py with next index
    
    return "end"          # No more URLs to try