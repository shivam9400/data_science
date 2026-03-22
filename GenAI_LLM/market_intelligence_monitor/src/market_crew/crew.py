import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool
from crewai import LLM

# Create a wrapper that CrewAI recognizes
@tool("duckduckgo_search")
def search_tool_fn(query: str):
    """Search the internet for information about tech products and pricing."""
    return DuckDuckGoSearchRun().run(query)

@CrewBase
class TechMarketCrew():
    """TechMarketCrew setup for Tech & Durables Analysis"""
    
    # Load our YAML configurations
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        # Initialize the Free Gemini Model
        self.llm = LLM(
            model="gemini/gemini-3.1-flash-lite-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Initialize the Free Search Tool
        self.search_tool = DuckDuckGoSearchRun()

    @agent
    def technical_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_researcher'],
            tools=[search_tool_fn],
            llm=self.llm,
            verbose=True
        )

    @agent
    def market_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['market_strategist'],
            llm=self.llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @task
    def strategic_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['strategic_analysis_task'],
            output_file = "output/market_report.md"
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential, # Researcher -> Strategist
            verbose=True
        )