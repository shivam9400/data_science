# This is a foundational, runnable example of an agentic AI framework.
# It simulates the core loop of an agent: perceive, decide, and act.
# In a real-world application, the `decide` method would typically involve
# a call to a Large Language Model (LLM).

import time
import json
import random

# A simple tool to simulate a web search, as per user instructions.
# In a real application, this would be a more complex API call.
# The search tool is included to demonstrate how an agent uses external functionalities.
# It uses a placeholder tool call for demonstration purposes.
def google_search(query):
    """Simulates a Google Search API call."""
    print(f"Executing search for: '{query}'...")
    # This is a placeholder for the actual tool call.
    # In a real system, this would make an API call and return the result.
    # We will simulate a search result based on the query.
    if "latest news" in query:
        return "Search result: 'AI continues to advance, with new models announced weekly.'"
    if "weather" in query:
        return "Search result: 'The weather is sunny with a high of 75°F.'"
    return f"Search result for '{query}': No specific information found."

class Tool:
    """
    Represents an external tool an agent can use.

    Attributes:
        name (str): The name of the tool.
        description (str): A description of what the tool does.
        function (callable): The function to execute when the tool is used.
    """
    def __init__(self, name, description, function):
        self.name = name
        self.description = description
        self.function = function

class Environment:
    """
    Represents the world the agent interacts with.

    Attributes:
        state (str): The current state of the environment.
        tools (dict): A dictionary mapping tool names to Tool objects.
    """
    def __init__(self):
        self.state = "The agent is in a quiet room."
        self.tools = {
            "google_search": Tool(
                name="google_search",
                description="A tool to search for information on the web.",
                function=google_search
            )
        }

    def update_state(self, new_state):
        """Updates the environment's state."""
        self.state = new_state

    def execute_action(self, action):
        """
        Executes an action in the environment.
        An action can be a simple state change or a tool use.
        """
        if isinstance(action, tuple) and action[0] in self.tools:
            tool_name, tool_input = action
            tool_result = self.tools[tool_name].function(tool_input)
            self.update_state(f"The agent used the {tool_name} tool. Result: {tool_result}")
        else:
            self.update_state(action)

class Agent:
    """
    The core agent class that perceives, decides, and acts.

    Attributes:
        name (str): The name of the agent.
        tools (list): A list of Tool objects the agent has access to.
    """
    def __init__(self, name, tools=None):
        self.name = name
        self.tools = tools if tools is not None else []
        self.memory = []
        print(f"Agent '{self.name}' initialized with {len(self.tools)} tools.")

    def perceive(self, environment):
        """
        Observes the current state of the environment.
        In a real application, this could involve processing sensor data.
        """
        perception = environment.state
        print(f"Agent '{self.name}' perceives: '{perception}'")
        return perception

    def decide(self, perception):
        """
        Decides on the next action based on perception and memory.
        This is the brain of the agent. In a real system, an LLM would generate the action.
        """
        print(f"Agent '{self.name}' is deciding on the next action...")
        self.memory.append(perception)

        # Simple decision-making logic based on keywords.
        # This is where an LLM call would be made in a real agentic framework.
        if "time to search" in perception:
            # Decide to use a tool.
            return ("google_search", "latest news on AI")
        elif "something is not right" in perception:
            # Decide to use a tool with a different query.
            return ("google_search", "current weather")
        elif len(self.memory) > 3 and "search" in self.memory[-2]:
            # Decide to take a different action after using a tool.
            return f"The agent is satisfied with the search results and is now thinking."
        else:
            # Decide on a simple action.
            return f"The agent is contemplating its existence."

    def act(self, action, environment):
        """
        Executes the chosen action in the environment.
        """
        print(f"Agent '{self.name}' is acting: '{action}'")
        environment.execute_action(action)

def main():
    """Main function to run the agentic framework simulation."""
    print("--- Starting Agentic Framework Simulation ---")
    
    # 1. Initialize the environment and tools.
    env = Environment()
    
    # 2. Initialize the agent with its tools.
    tools = [env.tools["google_search"]]
    agent = Agent(name="Gemini-AI", tools=tools)

    # 3. Simulation loop.
    steps = 0
    while steps < 5:
        print("\n--- Step", steps + 1, "---")
        
        # 3a. Agent perceives the environment.
        perception = agent.perceive(env)
        
        # 3b. Agent decides on an action.
        # We'll manually change the environment state to trigger a tool use.
        if steps == 1:
            env.update_state("The agent feels like it's time to search for information.")
        elif steps == 3:
            env.update_state("The agent feels like something is not right, and decides to check weather.")
        
        action = agent.decide(perception)
        
        # 3c. Agent acts on the environment.
        agent.act(action, env)

        steps += 1
        time.sleep(1) # Pause for 1 second to make the simulation readable.

    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    main()
    #print(2+2)
