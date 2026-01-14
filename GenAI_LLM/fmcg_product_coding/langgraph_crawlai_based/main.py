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
        "current_url_index": 0,
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
    prod_desc = "Nivea Intense Repair Shampoo 250 ml"
    #prod_desc = "Hộp 30 Viên Giặt Xả Quần Áo Cao Cấp 3IN1 sử dụng tiện lợi, dễ dàng siêu sạch thơm lừng, Viên Giặt Xả 0 Quần Áo Đa Năng 3IN1 Siêu Tiện Lợi Hộp 30 Viên Giúp Giặt Quần Áo Dễ Dàng Tiện Lợi Cực Sạch Và Thơm"
    
    pred_cat = "Shampoo"
    pred_brand = "Nivea"
    asyncio.run(run_coder(prod_desc))