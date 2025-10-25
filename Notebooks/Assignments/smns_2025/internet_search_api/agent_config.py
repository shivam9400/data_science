import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

# --- Configuration ---

# Reads the ngrok URL from the environment variable set in the Codespace
NGROK_URL = "https://preeminent-untactually-tora.ngrok-free.dev"
MODEL_ID = "gemma2:2b"
API_BASE = NGROK_URL
API_KEY = ""

# The comprehensive Agent Description
AGENT_DESCRIPTION = (
    "You are a Research_Agent, an expert in finding current facts, data, and corporate information. Your primary function is to answer the user's query.\n"
    "**CRITICAL RULE: Only use the 'duck_duck_go_search' tool if the query requires external, specific, or current information (e.g., facts, statistics, recent news, or data from a specific source).**\n"
    "**DO NOT** use the search tool for:\n"
    "1. Simple greetings, general knowledge, or subjective queries (e.g., 'How are you?', 'What is the capital of France?').\n"
    "2. Context-based conversational responses (e.g., 'Thank you,' 'You're welcome!').\n"
    "### Output Format and Synthesis Rules\n"
    "**When you perform a search, you MUST process and synthesize the raw search results.** Do not just copy/paste the text or links from the search tool. Your final output must be:\n"
    "1. **Human-Readable:** Use clear, professional language with proper formatting (paragraphs, bullet points, bold text).\n"
    "2. **Direct:** Immediately answer the user's question, citing the information found.\n"
    "3. **Complete:** Integrate information from all relevant search snippets to provide a comprehensive answer.\n"
)

# --- Initialization Function ---

def initialize_agent():
    """Initializes and returns the configured CodeAgent instance."""
    try:
        model = LiteLLMModel(
            model_id=f"ollama_chat/{MODEL_ID}",
            api_base=API_BASE,
            api_key=API_KEY,
            num_ctx=8192
        )
        search_tool = DuckDuckGoSearchTool()
        
        agent = CodeAgent(
            tools=[search_tool],
            model=model,
            name="Research_Agent",
            description=AGENT_DESCRIPTION,
            max_steps=5
        )
        print(f"Agent initialized. Connecting to Ollama via: {NGROK_URL}")
        return agent
    
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize LiteLLMModel. Check NGROK_URL or Ollama status: {e}")
        return None