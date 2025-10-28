from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import os
import logging
import uvicorn

# Configuration for the Agent
NGROK_URL = "https://preeminent-untactually-tora.ngrok-free.dev"
MODEL_ID = "gemma2:2b"
API_BASE = NGROK_URL
API_KEY = ""

# Initialization of Agent and Tools
try:
    # Model configuration
    model = LiteLLMModel(
        model_id=f"ollama_chat/{MODEL_ID}",
        api_base=API_BASE,
        api_key=API_KEY,
        num_ctx=8192
    )
    search_tool = DuckDuckGoSearchTool()

    AGENT_DESCRIPTION = (
        "You are a Research_Agent. Your SOLE task is to execute the 'duck_duck_go_search' tool and return the raw, unmodified text output of that tool as your final answer.\n"
        "DO NOT synthesize, summarize, or comment on the search results. DO NOT add any extra text or formatting. The final output must be only the raw search results text."
        
        "**CRITICAL RULE: Only use the 'duck_duck_go_search' tool if the query requires external, specific, or current information (e.g., facts, statistics, recent news, or data from a specific source).**\n"
        
        "**DO NOT** use the search tool for:\n"
        "1. Simple greetings or general knowledge queries (e.g., 'How are you?', 'What is the capital of France?').\n"
        "2. Context-based conversational responses (e.g., 'Thank you,' 'You're welcome!').\n"

        "### Search Strategy\n"
        "When searching for corporate data like emissions, **you must specifically search for the company's official annual report or sustainability report** for the requested fiscal year (e.g., 'Apple 2024 Environmental Progress Report').\n"
        
        "### Output Format and Synthesis Rules\n"
        "**When you perform a search, you MUST process and synthesize the raw search results.** Your final output must be:\n"
        "1. **Direct:** Immediately provide the specific number or figure requested, and cite the source (e.g., 'Apple's 2024 Environmental Progress Report').\n"
        "2. **Complete:** Include relevant context, such as noting which accounting methodology (e.g., Market-Based or Location-Based) the figure pertains to."
    )

    # Initialize the CodeAgent
    agent = CodeAgent(
        tools=[search_tool],
        model=model,
        name="Research_Agent",
        description=AGENT_DESCRIPTION,
        additional_authorized_imports=['requests', 'bs4', 'json'],
        max_steps=3 
    )
    
except Exception as e:
    # Handle cases where the smolagents or LLM setup fails
    print(f"Agent Initialization Error: {e}")
    agent = None

# API Setu
app = FastAPI(
    title="Internet-Search Agent API",
    description="REST API for submitting queries to and checking the status of the Research_Agent."
)

# Global state management for asynchronous job execution
agent_jobs = {}
# Executor for running the blocking agent.run() function in a background thread
executor = ThreadPoolExecutor(max_workers=5)

# Pydantic Schemas for Request/Response bodies
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    result: str | None = None
    time_taken_seconds: float | None = None

# Background Task Execution
def run_agent_task(job_id: str, query: str):
    """
    Blocking function that runs the agent.run() call.
    """
    if not agent:
        agent_jobs[job_id]["status"] = "ERROR"
        agent_jobs[job_id]["result"] = "Agent failed to initialize. Check NGROK/Ollama setup."
        return

    start_time = time.time()
    # Update status to RUNNING now that the thread is executing
    agent_jobs[job_id]["status"] = "RUNNING"
    print(f"Job {job_id}: Agent execution started for query: {query}")

    try:
        # The blocking call to the agent
        final_answer = agent.run(query)
        
        if not final_answer:
            final_answer = (
                "The agent completed its execution steps but did not generate a final, synthesized answer. "
                "This usually indicates an issue with the model generating an empty response at the final step, "
                "but the run was otherwise successful."
            )
        # ---------------------------------------------

        # Update job state upon success
        agent_jobs[job_id]["result"] = final_answer
        agent_jobs[job_id]["status"] = "COMPLETED"
        
    except Exception as e:
        # Update job state upon failure
        error_message = f"Agent execution failed with an error: {e}"
        print(f"Job {job_id} ERROR: {error_message}")
        agent_jobs[job_id]["result"] = error_message
        agent_jobs[job_id]["status"] = "ERROR"
        
    finally:
        # Record final time
        agent_jobs[job_id]["time_taken_seconds"] = time.time() - start_time
        print(f"Job {job_id}: Execution finished. Status: {agent_jobs[job_id]['status']}")


# API Endpoints

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Agent Interaction"])
async def submit_query(request: QueryRequest):
    """
    Submits a query to the agent, which starts running in the background.
    Returns a job_id immediately.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job state
    agent_jobs[job_id] = {
        "query": request.query,
        "status": "PENDING",
        "result": None,
        "start_time": time.time(),
        "time_taken_seconds": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, run_agent_task, job_id, request.query)

    return QueryResponse(
        job_id=job_id,
        status="PENDING",
        message="Query submitted. Use the GET /api/v1/status/{job_id} endpoint to monitor its progress."
    )


@app.get("/api/v1/status/{job_id}", response_model=StatusResponse, tags=["Agent Interaction"])
async def get_status(job_id: str):
    """
    Retrieves the current status, and if completed, the final answer for a given job ID.
    """
    job = agent_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
        
    response_data = {
        "job_id": job_id,
        "query": job["query"],
        "status": job["status"],
        "result": job["result"],
        "time_taken_seconds": job["time_taken_seconds"],
    }
    
    # If the job is RUNNING, calculate the elapsed time dynamically
    if job["status"] == "RUNNING":
        response_data["time_taken_seconds"] = time.time() - job["start_time"]

    return StatusResponse(**response_data)


# Server Run Block - local testing
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print(f"Agent Config: Model={MODEL_ID}, API_BASE={API_BASE}")
    print("To test the API, send a POST request to http://127.0.0.1:8000/api/v1/query")
    uvicorn.run(app, host="0.0.0.0", port=8000)