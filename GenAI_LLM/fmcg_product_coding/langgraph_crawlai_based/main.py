'''
Uses Python's asyncio to handle the web crawling and LLM calls efficiently
'''

import asyncio
from dotenv import load_dotenv
from graph import app

load_dotenv()

async def run_coder(sku_string: str):
    initial_state = {
        "raw_input": sku_string,
        "found_urls": [],
        "scraped_content": "",
        "final_output": None,
        "error": None
    }
    
    # Execute the LangGraph
    final_state = await app.ainvoke(initial_state)
    
    if final_state['final_output']:
        print(f"--- SUCCESS: {sku_string} ---")
        print(final_state['final_output'].model_dump_json(indent=2))
    else:
        print(f"--- FAILED: {final_state['error']} ---")

if __name__ == "__main__":
    # Test with a messy FMCG string
    asyncio.run(run_coder("Niv Men Clsc 50ml"))
