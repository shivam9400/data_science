# Smartphone Market Intelligence Monitor

## Problem Statement
In the Indian Durables market, product lifecycles are short and price volatility is high. Manual tracking of competitor specs (RAM, ROM, CPU) and pricing is slow and prone to error.

## Solution
A multi-agent "Crew" that automates web research and strategic positioning. The system identifies "feature-price gaps" to help brands adjust their market strategy.

## Crew
1. Technical Researcher: Scrapes and tabulates specs.
2. Market Strategist: Analyzes data and suggests positioning.

### Tasks

| **Task** | **Description** | **Expected Output** |
|---|---|---|
| **Market Research** | Search for the 3 latest mid-range smartphones launched in India; Extract: Processor; RAM; ROM; Camera; Current Price | **A Markdown table comparing the 3 phones** |
| **Strategy Analysis** | Review the table; Identify which phone offers the best "Spec-to-Price" ratio; Suggest a counter-move for a rival brand | **A 3-point strategic memo with a SWOT analysis** |

## Key Features
1. Structured Markdown output, real-time price tracking, and strategic SWOT-style insights.
2. Configuration is decoupled from logic - by defining agents and tasks in YAML files, we can modify an agent's personality or a task'r requirements without changing the python code. This makes system much easier to tune and scale.
3. Structure follows the moder `crewai` recommendation:
    1. `crew.py` houses the definition of "Crew" (the team).
    2. `main.py` acts as the orchestrator to trigger the process.

## Important Points for CrewAI Framework
1. **Naming Consistency is Critical**: The `agent` property in `tasks.yaml` acts as a link to the `@agent` decorated function names in `crew.py`. The agent in tasks.yaml should match function name exactly. Additionally, the keys (agent names) defined in `agents.yaml` must match the strings used in the `config` parameter within `@agent`.
    1. Keys defined in `agents.yaml` can be anything, but should be called carefully while passing to config of `@agent`.