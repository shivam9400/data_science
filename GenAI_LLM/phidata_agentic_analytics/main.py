import pandas as pd
import duckdb
import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Groq API Call Functions ---

def get_chat_completion(prompt, api_key):
    """
    Sends a prompt to the Groq API and returns the generated text.
    """
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# --- Main Script ---

# 1. Load data from a CSV file
print("Loading data...")
try:
    df = pd.read_csv("https://raw.githubusercontent.com/berkyalkn/llm-sql-visualizer-notebook/refs/heads/main/netflix_titles.csv")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Connect to DuckDB and register the pandas DataFrame as a table
con = duckdb.connect(database=':memory:', read_only=False)
con.execute("CREATE TABLE df AS SELECT * FROM df")

# 2. Get the table schema
schema_query = "PRAGMA table_info('df')"
schema_df = con.query(schema_query).fetchdf()
table_schema = schema_df[['name', 'type']].to_string(index=False)
print("Table Schema:\n", table_schema)

# Choose a valid column for aggregation (fallback to 'title' if 'name' doesn't exist)
valid_columns = schema_df['name'].tolist()
agg_column = 'title' if 'title' in valid_columns else valid_columns[0]

# 3. Define the LLM prompt
prompt = f"""
You are an expert at converting English questions to SQL queries.
You will be provided with a dataframe and a question.
Your task is to generate a valid SQL query for the given question.
The table name is 'df'.
The table schema is:
{table_schema}
The question is: 'What are the top 5 customers with the most sales?'
Use the column '{agg_column}' for counting.
Return only the SQL query and nothing else.
Do not wrap the query in markdown.
"""

# 4. Generate the SQL query using the LLM
print("\nGenerating SQL query...")
try:
    sql_query = get_chat_completion(prompt, api_key)
    # Remove any extra characters or markdown
    sql_query = sql_query.strip().strip('`').replace("sql\n", "").replace("```", "")
    print(f"Generated Query:\n{sql_query}")
except Exception as e:
    print(f"Error generating query with Groq API: {e}")
    print("Please check your API key and network connection.")
    exit()

# 5. Execute the SQL query
print("\nExecuting query...")
try:
    result_df = con.query(sql_query).fetchdf()
    print("\nQuery Results:")
    print(result_df)
except duckdb.ParserException as e:
    print(f"Error parsing SQL query: {e}")
except duckdb.BinderException as e:
    print(f"Error binding SQL query: {e}")
except Exception as e:
    print(f"An unexpected error occurred during query execution: {e}")

# 6. Close the DuckDB connection
con.close()
print("\nScript finished.")
