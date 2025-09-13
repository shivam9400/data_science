import json
import tempfile
import csv
import streamlit as st
import pandas as pd
#from agno.models.openai import OpenAIChat
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from agno.tools.pandas import PandasTools
import re
from huggingface_hub import InferenceClient
from phi.model.base import Model, ModelResponse

from pydantic import BaseModel, PrivateAttr
from datetime import datetime

from huggingface_hub import HfApi

from phi.model.base import Model
from huggingface_hub import InferenceClient
import json


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



class MessageProxy:
    def __init__(self, content, role="assistant", audio=None, created_at=None):
        self.content = content
        self.role = role
        self.audio = audio
        self.created_at = created_at or datetime.utcnow()

# class HuggingFaceProxy:
#     def __init__(self, token, model="meta-llama/Meta-Llama-3-8B-Instruct"):
#         self.name = "llama3"
#         self.description = "Meta-Llama-3-8B-Instruct via Hugging Face Inference API"
#         self.client = InferenceClient(model=model, token=token)

#     def response(self, messages):
#         hf_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
#         response = self.client.chat_completion(hf_messages, max_tokens=512)
#         return MessageProxy(content=response.choices[0].message.content)

# class DuckDbModel(BaseModel):
#     name: str = "llama3"
#     description: str = "Meta-Llama-3-8B-Instruct via Hugging Face Inference API"
#     _client: InferenceClient = PrivateAttr()

#     def __init__(self, token, model="meta-llama/Meta-Llama-3-8B-Instruct"):
#         super().__init__()
#         self._client = InferenceClient(model=model, token=token)

#     def response(self, messages):
#         hf_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
#         response = self._client.chat_completion(hf_messages, max_tokens=512)
#         return MessageProxy(content=response.choices[0].message.content)
    
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
    openai_key = st.text_input("Enter your Hugging Face API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("Hugging Face API key saved!")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
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
            model=HuggingFaceModel(token=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[],
            markdown=True,
            system_prompt=(
                "You are an expert data analyst. Generate SQL queries to solve the user's query. "
                "Return only the SQL query, enclosed in ```sql``` and give the final answer."
            ),
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
                        stream=False,
                        )

                    # Display the response in Streamlit
                    st.markdown(response_content)
                
                    
                except Exception as e:
                    import traceback
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")
                    st.error(traceback.format_exc())