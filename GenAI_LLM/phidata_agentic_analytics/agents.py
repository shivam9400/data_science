from agno.agent import Agent
from agno.tools.duckdb import DuckDbTools
#from agno.storage.agent.sqlite import SqlAgentStorage
from agno.db.sqlite import SqliteDb
from agno.models.groq import Groq
#from agno.run.response import RunEvent, RunResponse
import os
from dotenv import load_dotenv
import duckdb
import pandas as pd

url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_rankings_current.csv"
df = pd.read_csv(url)

# Connect to DuckDB and create table
# con = duckdb.connect()
# con.execute("CREATE TABLE atp_rankings AS SELECT * FROM df")

# Load your GROQ API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the agent with Groq model
agent = Agent(
    tools=[DuckDbTools()],
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    #show_tool_calls=True,
    instructions=[
        "You can query the database tables using natural language. "
        "Translate the user's question into SQL and return the answer."
        """When running SELECT queries, make sure that you put all field names 
        in double quotes to avoid syntax errors.
        e.g. SELECT "column name" FROM "table_name" """
    ],
    #add_datetime_to_instructions=True,
    #add_history_to_messages=True,
    #storage=SqlAgentStorage(table_name="agent_sessions", db_file="tmp/agent.db"),
    #db=SqliteDb(db_file="tmp/agent.db")
)
agent.run(f"Load this CSV into a table called 'atp_rankings': {url}")

# Now query it in natural language
query = "What is the average points of players in the current ATP rankings?"
result = agent.run(query)
print(result.content)

# Streaming helper
def as_stream(response):
    for chunk in response:
        if isinstance(chunk, str):
                yield chunk

# for output in as_stream(agent.run("What is the average rating of movies?")):
#     print(output)

# agent_run = agent.run("What is the average rating of movies?", stream=False)
# avg_rating = con.execute('SELECT AVG("rating") FROM "movies"').fetchone()[0]
# result = agent.run(f"The average rating of movies is {avg_rating}.")
result = agent.run(f"I want to query this file --> https://raw.githubusercontent.com/JeffSackmann/tennis_atp/refs/heads/master/atp_rankings_current.csv")
print(result.content)