import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.agent.duckdb import DuckDbAgent
from huggingface_hub import InferenceClient
from phi.model.base import Model, ModelResponse
from pydantic import PrivateAttr
from phi.model.base import Model
from huggingface_hub import InferenceClient
import json
import matplotlib.pyplot as plt
import io
import duckdb
import re

class HuggingFaceModel(Model):
    # Define a required Pydantic field that `DuckDbAgent` expects
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Private (non-Pydantic) attribute
    _client: InferenceClient = PrivateAttr()

    def __init__(self, token: str, model: str = "meta-llama/Meta-Llama-3-8B-Instruct", **kwargs):
        # Pass the `model` field up to BaseModel
        super().__init__(model=model, **kwargs)

        # Attach Hugging Face client
        self._client = InferenceClient(model=model, token=token)

    def response(self, messages, **kwargs):
        hf_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = self._client.chat_completion(hf_messages, max_tokens=512)
        return ModelResponse(content=response.choices[0].message.content)
    
def visualize_data(df, x=None, y=None, kind="bar", title="Chart"):
    fig, ax = plt.subplots(figsize=(8, 5))

    if kind == "bar":
        df.plot.bar(x=x, y=y, ax=ax)
    elif kind == "line":
        df.plot.line(x=x, y=y, ax=ax)
    elif kind == "scatter":
        df.plot.scatter(x=x, y=y, ax=ax)
    else:
        st.error(f"Visualization type {kind} not supported.")
        return None

    ax.set_title(title)
    st.pyplot(fig)

def extract_sql(text: str) -> str:
    """
    Extract the first SQL query from model output.
    """
    # Look for SQL inside ```sql ... ```
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Otherwise, try to find a SELECT ... statement
    match = re.search(r"(SELECT .*?;)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: return the text as-is (but this may still fail)
    return text.strip()

def sanitize_sql(sql: str) -> str:
    """
    Fix common non-DuckDB SQL issues.
    """
    sql = sql.replace("show_tables()", "SHOW TABLES")
    sql = sql.replace("SHOW TABLES()", "SHOW TABLES")
    return sql

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    hugging_face_key = st.text_input("Enter your Hugging Face API key:", type="password")
    if hugging_face_key:
        st.session_state.hugging_face_key = hugging_face_key
        st.success("Hugging Face API key saved!")
    else:
        st.warning("Please enter your hugging_face API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "hugging_face_key" in st.session_state:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table
        
        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)
        
        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }
        
        # Initialize the DuckDbAgent for SQL query generation
        duckdb_agent = DuckDbAgent(
            model=HuggingFaceModel(token=st.session_state.hugging_face_key),
            semantic_model=json.dumps(semantic_model),
            tools=[],
            markdown=True,
            system_prompt=(
                "You are an expert data analyst. "
                "You can only generate valid DuckDB SQL. "
                "Important rules: "
                "- Use `SHOW TABLES;` instead of `show_tables()` "
                "- Do not use SQL dialect from Postgres/MySQL. "
                "- Always wrap SQL queries in ```sql fences. "
                "If the user asks for visualization, include JSON with chart details."
                )
                )
        # Initialize code storage in session state
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = None
        
        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")
        
        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")
        
        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get the response from DuckDbAgent
               
                        response1 = duckdb_agent.run(user_query)

                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        response = duckdb_agent.print_response(
                            user_query,
                            stream=False
                            )
                                                

                    # Display the response in Streamlit
                    st.markdown("### Agent Response")
                    st.markdown(response_content)
                  
                    con = duckdb.connect()
                    con.execute(f"CREATE OR REPLACE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{temp_path}')")

                    st.markdown("### clean SQL")
                    clean_sql = extract_sql(response_content)
                    st.markdown(clean_sql)
                    #clean_sql = sanitize_sql(clean_sql)
                    st.markdown("### SQL Executed")
                    st.code(clean_sql, language="sql")
                    query_result = con.execute(clean_sql).df()
                    st.dataframe(query_result)

                    if "bar chart" in user_query.lower():
                        visualize_data(query_result, x=query_result.columns[0], y=query_result.columns[1], kind="bar")
                    elif "line chart" in user_query.lower():
                        visualize_data(query_result, x=query_result.columns[0], y=query_result.columns[1], kind="line")
                
                except Exception as e:
                    import traceback
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")
                    st.error(traceback.format_exc())