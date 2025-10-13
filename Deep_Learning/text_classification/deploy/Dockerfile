# Use official python runtime
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# This copies everything from your project root into /app
COPY . .

# Ensure the artifacts directory is present (if you're not downloading the model)
RUN mkdir -p artifacts

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
