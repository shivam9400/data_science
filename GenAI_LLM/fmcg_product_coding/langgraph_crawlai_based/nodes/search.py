'''  Connects to your local SearXNG Docker container  '''

import requests
import os

def get_url_quality_score(url, product_description):
    score = 0
    url_lower = url.lower()
    prod_desc_parts = product_description.lower().split()

    # 1. Product Detail Page (PDP) Indicators (+20 points)
    pdp_patterns = ['/dp/', '/p/', '/product/', '/item/', '-pid-', '.html']
    if any(p in url_lower for p in pdp_patterns):
        score += 20

    # 2. Keyword Match in URL (+10 points per keyword)
    # If "nivea" and "shampoo" are in the URL string, it's highly relevant
    for part in prod_desc_parts:
        if len(part) > 2 and part in url_lower:
            score += 10

    # 3. Noise Penalties (-50 points)
    # If the URL contains "search", "category", or "collections", it's a grid page
    noise_patterns = ['/search', '/category', '/collections', 'filter=', '?s=', '/browse']
    if any(p in url_lower for p in noise_patterns):
        score -= 50

    return score

def search_node(state):
    query = f"{state['raw_input']} specifications ingredients directions"
    response = requests.get(
        f"http://localhost:8080/search",
        params={"q": query, "format": "json"}
    )

    raw_results = response.json().get('results', [])
    
    filtered_urls = []
    for result in raw_results:
        url = result['url'].lower()
        
        # Optimization: Exclude typical "Noise" URLs
        noise_patterns = ['/search', '/category', '/collections', '/tags', 'filter=']
        if any(pattern in url for pattern in noise_patterns):
            continue
            
        # Optimization: Prioritize Product Detail Pages (PDP)
        # Most e-commerce sites use /p/ /dp/ or /product/ for single items
        if any(p in url for p in ['/p/', '/dp/', '/product/', '.html']):
            filtered_urls.append(result['url'])

    scored_urls = []
    for url in filtered_urls:
        quality_score = get_url_quality_score(url, state['raw_input'])
        scored_urls.append({
            "url": url,
            "score": quality_score
        })
    scored_urls.sort(key=lambda item: item['score'], reverse=True)
    best_urls = [c['url'] for c in scored_urls]

    return {"found_urls": best_urls[:2]}