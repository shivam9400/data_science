from dotenv import load_dotenv
from src.market_crew.crew import TechMarketCrew

# Environment setup: Load API Keys from .env
load_dotenv()

if __name__ == "__main__":
    # Initialization: Create an instance of your custom Crew defined in the src folder
    test_crew_instance = TechMarketCrew()
    
    # Input Parameters: These values are passed into the YAML templates
    inputs = {
        'category': 'Smartphones',
        'reference_product': 'iPhone 15'
    }
    
    print("\n--- Kicking off the Crew ---")
    try:
        # Execution: .crew() assembles the agents and tasks, .kickoff() starts the process
        result = test_crew_instance.crew().kickoff(inputs=inputs)
        print(f"Final Result: {result}")
    except Exception as e:
        print(f"Crew failed: {e}")