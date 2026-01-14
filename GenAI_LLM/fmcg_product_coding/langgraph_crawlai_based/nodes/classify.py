import os
import json
from openai import AzureOpenAI, OpenAI
from schema import FMCGProduct
from pydantic import ValidationError
import re

# Set this at the top for easy switching
MODEL_TYPE = "ollama" 

if MODEL_TYPE == "ollama":
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
else:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-08-06"
    )

def classify_node(state):
    # PROMPT: Includes a one-shot example to force flat JSON structure
    system_prompt = """
    You are a ReAct Agent. Your goal is to extract product attributes into a single FLAT JSON object.
    THOUGHT: Look for category clues in product rankings (e.g., "# in Hair Shampoo") or breadcrumbs.
    ACTION: Extract the FLAT JSON.
    
    CRITICAL RULES:
    1. Output ONLY a flat JSON object. No nesting.
    2. 'brand' must be a simple STRING (e.g., "Nivea"), not an object.
    3. If information is missing, use "Unknown" for strings and 0.0 for confidence.
    4. If the content is a BOT CHECK, CAPTCHA, or 'No featured offers', output: {"action": "RETRY"}
    5. If the content is high quality, extract the data into a flat JSON and output: {"action": "FINALIZE", "data": {...}}

    EXAMPLE OUTPUT:
    {
    "action": "FINALIZE", 
    "data" : 
    {
    "brand": "Nivea",
    "category": "Deodorant",
    "confidence": 0.95
    }
    """
    clean_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', state['scraped_content'])

    if MODEL_TYPE == "ollama":
        response = client.chat.completions.create(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Description: {state['raw_input']}\nContent: {clean_content[:3000]}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        # --- ROBUST PARSING LOGIC ---
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
            
            if data.get("action") == "RETRY":
                # Increment index to try next URL in next crawl_node call
                return {
                    "current_url_index": state["current_url_index"] + 1,
                    "thought": "Content was blocked or insufficient. Trying next URL.",
                    "scraped_content": "" # Clear junk content
                    }
    
            # 1. Handle "Container Nesting" (if model wraps everything in 'product' or 'data')
            if "product" in data: data = data["product"]
            elif "data" in data: data = data["data"]

            # 2. Fix nested "brand" error
            if isinstance(data.get("brand"), dict):
                data["brand"] = data["brand"].get("full_name") or data["brand"].get("name") or "Unknown"

            # 3. Ensure mandatory fields exist (defaulting to "Unknown" to prevent crashes)
            required_fields = ["brand", "category", "sub_category", "pack_size"]
            for field in required_fields:
                if field not in data:
                    data[field] = "Unknown"

            # 4. Force confidence to float
            try:
                data["confidence"] = float(data.get("confidence", 0.0))
            except:
                data["confidence"] = 0.0

            # Final Pydantic Validation
            final_record = FMCGProduct(**data)
            return {"final_output": final_record}

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"ERROR: Raw model output was: {raw_content}")
            # Fallback record so the graph doesn't crash
            return {"final_output": FMCGProduct(brand="Error", category="Error", sub_category="Error", pack_size="N/A", confidence=0.0)}
    
    else:
        # Azure logic remains the same (it handles validation natively)
        completion = client.beta.chat.completions.parse(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SKU: {state['raw_input']}\nContext: {state['scraped_content'][:5000]}"},
            ],
            response_format=FMCGProduct,
        )
        return {"final_output": completion.choices[0].message.parsed}