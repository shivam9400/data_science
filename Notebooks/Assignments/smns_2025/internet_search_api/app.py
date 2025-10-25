import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from threading import Thread, Lock
import time
import random

# --- Pydantic Models for Request/Response Validation ---
class QueryRequest(BaseModel):
    """Model for the incoming query request."""
    query: str = Field(..., example="What is the latest market cap for Google?")

class TaskResponse(BaseModel):
    """Model for the response immediately after task submission."""
    status: str = Field(..., example="STARTING")
    message: str = Field(..., example="Task started asynchronously. Check the status endpoint for the result.")

class StatusResponse(BaseModel):
    """Model for the status retrieval response."""
    status: str = Field(..., example="IDLE")
    message: str = Field(None, example="Agent is currently processing the query.")
    result: str = Field(None, example="The agent found the requested data.")

# --- Simulated Agent Execution Function ---
def mock_agent_run(query: str):
    """Simulates a long-running search agent task."""
    # Simulate a time-consuming task
    time.sleep(random.randint(4, 7)) 
    
    # Simulate a potential failure based on query content
    if "fail" in query.lower():
        raise Exception("Simulated Agent Failure")

    return f"Search result for '{query}': The simulated search completed successfully after a delay."


# --- Agent Service for Managing State (Thread-Safe Status Tracking) ---
class AgentService:
    def __init__(self):
        self.status = "IDLE"
        self.lock = Lock()
        self.result = None
        self.thread = None

    def run_query_async(self, query: str):
        with self.lock:
            if self.status != "IDLE":
                raise HTTPException(status_code=409, detail=f"Agent is busy. Status: {self.status}")
            
            self.status = "STARTING"
            self.result = None
            
            # Use a thread to run the agent so the API request doesn't timeout
            self.thread = Thread(target=self._run_task, args=(query,))
            self.thread.start()

            return {"status": self.status, "message": "Task started asynchronously. Check the status endpoint for the result."}

    def _run_task(self, query: str):
        """Internal worker function executed in the background thread."""
        try:
            with self.lock:
                self.status = "RUNNING"
            
            # Execute the simulated agent function
            final_answer = mock_agent_run(query)

            with self.lock:
                self.result = final_answer
                self.status = "COMPLETED"
        
        except Exception as e:
            error_message = f"Agent failed during execution: {e}"
            print(f"Error: {error_message}")
            with self.lock:
                self.result = error_message
                self.status = "FAILED"
        
        finally:
            # Clean up the thread object after completion/failure
            time.sleep(1) 

    def get_status(self):
        """Retrieves the current status and result, resetting state if COMPLETED or FAILED."""
        with self.lock:
            if self.status in ["COMPLETED", "FAILED"]:
                temp_status = self.status
                temp_result = self.result
                
                # Reset the service state
                self.status = "IDLE" 
                self.result = None 
                self.thread = None
                
                return {"status": temp_status, "result": temp_result}
            
            return {"status": self.status, "message": "Agent is currently processing the query."}


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Minimal Agent Status API",
    description="A simple API demonstrating asynchronous query submission and status retrieval.",
    version="1.0"
)

# Initialize the service instance
agent_service = AgentService()


# --- API Endpoints ---
@app.post("/api/v1/query", response_model=TaskResponse, summary="Send a query to the agent")
async def send_query(request: QueryRequest):
    """Submits a query and starts the agent task in a background thread."""
    try:
        return agent_service.run_query_async(request.query)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/api/v1/status", response_model=StatusResponse, summary="Retrieve the agent's current status and result")
async def get_agent_status():
    """Retrieves the current status. Returns the result if COMPLETED or FAILED."""
    try:
        return agent_service.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

# --- Running the API ---
if __name__ == "__main__":
    # Standard way to run FastAPI with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)