import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool
from crewai import LLM

# CUSTOM TOOL: A wrapper for DuckDuckGo search that the agents can use
@tool("duckduckgo_search")
def search_tool_fn(query: str):
    """Search the internet for information about tech products and pricing."""
    return DuckDuckGoSearchRun().run(query)

@CrewBase
class TechMarketCrew():
    """Crew setup for Smartphone Analysis"""
    
    # Paths to the configuration files that define what agents/tasks do
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        # CONFIGURATION: Initialize the LLM (Gemini) and search tools
        self.llm = LLM(
            model="gemini/gemini-3.1-flash-lite-preview",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    # AGENT DEFINITION: The Technical Researcher - responsible for data gathering
    @agent
    def technical_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_researcher'],
            tools=[search_tool_fn],
            llm=self.llm,
            verbose=True
        )

    # AGENT DEFINITION: The Market Strategist - responsible for synthesis and drafting
    @agent
    def market_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['market_strategist'],
            llm=self.llm,
            verbose=True
        )

    # AGENT DEFINITION: The Trend Forecaster - responsible for predicting market movements
    @agent
    def trend_forecaster(self) -> Agent:
        return Agent(
            config=self.agents_config['trend_forecaster'],
            tools=[search_tool_fn],
            llm=self.llm,
            verbose=True
        )

    # TASK DEFINITION: The research phase
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    # TASK DEFINITION: The analysis and reporting phase
    @task
    def strategic_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['strategic_analysis_task'],
        )

    # TASK DEFINITION: The trend forecasting phase
    @task
    def trend_forecasting_task(self) -> Task:
        return Task(
            config=self.tasks_config['trend_forecasting_task'],
        )

    # WORKFLOW ASSEMBLY: Defines how the agents and tasks work together
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,         # Automatically collects all @agent functions
            tasks=self.tasks,           # Automatically collects all @task functions
            process=Process.sequential, # Flow: Researcher finishes -> Strategist starts
            verbose=True
        )