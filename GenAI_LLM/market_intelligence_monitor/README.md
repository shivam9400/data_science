# TechMarket-CrewAI

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
Structured Markdown output, real-time price tracking, and strategic SWOT-style insights.