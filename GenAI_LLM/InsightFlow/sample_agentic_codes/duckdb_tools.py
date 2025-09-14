from phi.agent import Agent
from phi.tools.duckdb import DuckDbTools
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
import duckdb

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

con = duckdb.connect()      # create duckdb connection
# load csv in movies table
con.execute("""
    CREATE OR REPLACE TABLE movies AS
    SELECT * FROM read_csv_auto('https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv')
""")

agent = Agent(
    name="DuckDB Agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    tools=[DuckDbTools(connection=con)],
    show_tool_calls=True,
    system_prompt="Use this file for Movies data: https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
)
agent.print_response("What is the average rating of movies?", markdown=True, stream=False)