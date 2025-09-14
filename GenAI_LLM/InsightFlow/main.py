import json
import csv
import pandas as pd
import re
from huggingface_hub import InferenceClient
from datetime import datetime
import os
from dotenv import load_dotenv

class MessageProxy:
    def __init__(self, content, role="assistant", audio=None, created_at=None):
        self.content = content
        self.role = role
        self.audio = audio
        self.created_at = created_at or datetime.utcnow()

class HuggingFaceProxy:
    def __init__(self, token, model="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.name = "llama3"
        self.description = "Meta-Llama-3-8B-Instruct via Hugging Face Inference API"
        self.client = InferenceClient(model=model, token=token)

    def response(self, messages):
        hf_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        response = self.client.chat_completion(hf_messages, max_tokens=512)
        return MessageProxy(content=response.choices[0].message.content)

# Function to preprocess and save the uploaded file

def preprocess_and_save(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, na_values=['NA', 'N/A', 'missing'])
        else:
            print("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        temp_path = file_path  # Use original file path for local script
        return temp_path, df.columns.tolist(), df
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None, None


def main():
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    file_path = os.getenv("DATA_FILE_PATH")
    if not token:
        print("HUGGINGFACE_TOKEN not found in .env file. Please add your token to .env.")
        return
    if not file_path:
        print("DATA_FILE_PATH not found in .env file. Please add your file path to .env.")
        return
    temp_path, columns, df = preprocess_and_save(file_path)
    if temp_path and columns and df is not None:
        print("Uploaded Data:")
        print(df.head())
        print("Uploaded columns:", columns)
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                }
            ]
        }
        agent = HuggingFaceProxy(token=token)
        while True:
            user_query = input("Ask a query about the data (or type 'exit' to quit): ").strip()
            if user_query.lower() == 'exit':
                break
            messages = [
                {"role": "user", "content": user_query}
            ]
            try:
                response = agent.response(messages)
                print("Agent response:")
                print(response.content)
            except Exception as e:
                print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
