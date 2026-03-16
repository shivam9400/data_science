from crewai import Agent, LLM
from langchain_ollama import ChatOllama
import os

# Initialize the local LLM
# Ensure you have run: ollama pull llama3.2
os.environ["OPENAI_API_KEY"] = "NA"

llm = ChatOllama(
    model="llama3.2:latest",
    base_url="http://localhost:11434/v1"
)

class ContentAgents:
    def strategist_agent(self):
        return Agent(
            role='Content Strategist',
            goal='Create a high-level content plan and blog outline for {topic}',
            backstory="""You are a veteran digital marketer. You excel at taking a 
            broad idea and breaking it down into a structured, SEO-friendly 
            blog outline that provides genuine value to the reader.""",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=False
        )

    def blogger_agent(self):
        return Agent(
            role='Professional Blogger',
            goal='Write a complete, engaging blog post based on a provided outline',
            backstory="""You are a specialized tech and lifestyle writer. Your 
            tone is conversational yet authoritative. You know how to use 
            Markdown formatting to make articles easy to read.""",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=False
        )

    def social_media_agent(self):
        return Agent(
            role='Social Media Manager',
            goal='Create catchy Instagram captions and image prompts for {topic}',
            backstory="""You are a creative genius who knows how to stop the scroll. 
            You generate punchy captions with relevant hashtags and provide 
            highly detailed visual descriptions for the Design team.""",
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=False
        )