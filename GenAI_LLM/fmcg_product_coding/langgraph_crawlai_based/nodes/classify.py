import os
import json
from openai import AzureOpenAI, OpenAI
from schema import FMCGProduct
from pydantic import ValidationError

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
    # REFINED PROMPT: Includes a one-shot example to force flat JSON structure
    system_prompt = """
    You are a strict FMCG Data Extraction API. Your goal is to extract product attributes into a single FLAT JSON object.
    
    CRITICAL RULES:
    1. Output ONLY a flat JSON object. No nesting.
    2. 'brand' must be a simple STRING (e.g., "Nivea"), not an object.
    3. If information is missing, use "Unknown" for strings and 0.0 for confidence.

    EXAMPLE OUTPUT:
    {
      "brand": "Nivea",
      "category": "Personal Care",
      "sub_category": "Deodorant",
      "pack_size": "50ml",
      "confidence": 0.95
    }
    """

    if MODEL_TYPE == "ollama":
        response = client.chat.completions.create(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"SKU: {state['raw_input']}\nContent: {state['scraped_content'][:3000]}"},
            ],
            response_format={"type": "json_object"} 
        )

        # --- ROBUST PARSING LOGIC ---
        raw_content = response.choices[0].message.content
        try:
            data = json.loads(raw_content)
            
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