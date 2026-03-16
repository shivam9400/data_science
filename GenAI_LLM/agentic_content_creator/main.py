import os
from datetime import datetime
from src.agents import ContentAgents, llm
from src.tasks import ContentTasks
from crewai import Crew, Process

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3.2:latest"
os.environ["OPENAI_API_KEY"] = "ollama"

def run_content_factory():
    # 1. Initialize Agents and Tasks
    agents = ContentAgents()
    tasks = ContentTasks()

    # Get user input
    # topic = input("Enter the topic for your content pack: ")
    topic = "bike ride near beach"

    # Create a unique folder for this project
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"content_vault/{timestamp}_{topic.replace(' ', '_')[:20]}"
    os.makedirs(folder_name, exist_ok=True)

    # 2. Define the Agent Team
    strategist = agents.strategist_agent()
    blogger = agents.blogger_agent()
    social_manager = agents.social_media_agent()

    # 3. Define the Workflow
    # Task 1: Create Outline -> Task 2: Write Blog -> Task 3: Social Media
    task1 = tasks.blog_task(strategist, topic, folder_name)
    task2 = tasks.writing_task(blogger, folder_name)
    task3 = tasks.social_task(social_manager, topic, folder_name)

    # 4. Kickoff the Crew
    crew = Crew(
        agents=[strategist, blogger, social_manager],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        manager_llm=llm,
        function_calling_llm=llm,
        planning_llm=llm,
        memory=False,
        verbose=True
    )

    print(f"\n🚀 Starting the Content Factory for: {topic}...\n")
    result = crew.kickoff()

    print("\n" + "="*50)
    print("FINAL TASK COMPLETED:")
    print(result)
    print("="*50)

    print(f"\n✅ Done! All files (Outline, Blog, Social) are saved in: {folder_name}")

    print(f"\n✅ Done! Your content pack is ready in: {folder_name}")

if __name__ == "__main__":
    run_content_factory()