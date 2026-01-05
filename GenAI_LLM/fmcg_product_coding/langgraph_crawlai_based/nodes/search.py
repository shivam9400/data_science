'''  Connects to your local SearXNG Docker container  '''

import requests
import os

def search_node(state):
    query = f"{state['raw_input']} product information brand category"
    response = requests.get(
        f"http://localhost:8080/search",
        params={"q": query, "format": "json"}
    )
    urls = [result['url'] for result in response.json().get('results', [])[:2]]
    return {"found_urls": urls}