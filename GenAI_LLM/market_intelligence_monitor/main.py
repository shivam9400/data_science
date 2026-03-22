from dotenv import load_dotenv
from src.market_crew.crew import TechMarketCrew

# Load API Keys from .env
load_dotenv()

def run():
    inputs = {
        'category': 'Premium Smartphones',
        'reference_product': 'iPhone 15 Pro'
    }
    TechMarketCrew().crew().kickoff(inputs=inputs)

# if __name__ == "__main__":
#     run()

if __name__ == "__main__":
    test_crew_instance = TechMarketCrew()
    
    # We define the inputs our YAML files are looking for
    inputs = {
        'category': 'Smartphones',
        'reference_product': 'iPhone 15'
    }
    
    print("\n--- Kicking off the Crew ---")
    try:
        result = test_crew_instance.crew().kickoff(inputs=inputs)
        print(f"Final Result: {result}")
    except Exception as e:
        print(f"Crew failed: {e}")