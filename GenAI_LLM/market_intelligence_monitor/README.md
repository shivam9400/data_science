# Smartphone Market Intelligence Monitor

## Problem Statement
In the Indian smartphone market, brands face multiple challenges:
- **Short product lifecycles** and **high price volatility** require constant market monitoring
- Manual competitive tracking (specs, pricing, market trends) is slow, error-prone, and fails to provide holistic insights
- Competitive intelligence is siloed: specs in one place, pricing elsewhere, sentiment scattered across multiple platforms
- Strategic decisions lack financial grounding—no quantified ROI scenarios for different positioning strategies
- Market trend analysis requires synthesizing data from multiple sources with no automated pipeline

## Solution
A 7-agent CrewAI system that automates comprehensive market intelligence gathering and analysis. Agents work sequentially, with each building upon the previous agent's insights through a file-based information pipeline. The system delivers competitive positioning analysis, pricing strategies, trend forecasts, and ROI scenarios—enabling data-driven decision making without manual research.

### Crew

A specialized team of 7 AI agents working together to provide comprehensive market intelligence:

1. **Technical Researcher** - Extracts specifications from latest product launches
2. **Market Strategist** - Compares specs and identifies value gaps vs. reference product
3. **Trend Forecaster** - Predicts 6-12 month market trends and technology adoption
4. **Price Analyst** - Tracks price movements and analyzes price elasticity
5. **Sentiment Analyst** - Analyzes consumer sentiment from reviews and social media
6. **Market Share Analyst** - Estimates market share and competitive positioning
7. **ROI Simulator** - Models financial scenarios and ROI impact of strategies

### Tasks

| **Agent** | **Task** | **Inputs** | **Expected Output** |
|---|---|---|---|
| **Technical Researcher** | Extract latest product specs and pricing | `{category}` | Specs table: RAM, ROM, Processor, Camera, Price |
| **Market Strategist** | Compare products vs. reference & SWOT analysis | Research data, `{reference_product}` | Strategic recommendations & positioning analysis |
| **Trend Forecaster** | Predict market trends and technology adoption | Research data, Strategic analysis | 6-12 month trend forecast |
| **Price Analyst** | Analyze pricing and price elasticity | Research data | Price history, elasticity scenarios, pricing strategy |
| **Sentiment Analyst** | Analyze consumer perception and sentiment | Research data | Sentiment breakdown, perception gaps by attribute |
| **Market Share Analyst** | Estimate market share and competitive trends | Research data | Market share estimates, competitive velocity |
| **ROI Simulator** | Model financial scenarios and ROI impact | All previous analysis | ROI scenarios, financial projections, breakeven analysis |

## Key Features
1. Structured Markdown output, real-time price tracking, and strategic SWOT-style insights.
2. Configuration is decoupled from logic - by defining agents and tasks in YAML files, we can modify an agent's personality or a task'r requirements without changing the python code. This makes system much easier to tune and scale.
3. Structure follows the moder `crewai` recommendation:
    1. `crew.py` houses the definition of "Crew" (the team).
    2. `main.py` acts as the orchestrator to trigger the process.

## Important Points for CrewAI Framework
1. **Naming Consistency is Critical**: The `agent` property in `tasks.yaml` acts as a link to the `@agent` decorated function names in `crew.py`. The agent in tasks.yaml should match function name exactly. Additionally, the keys (agent names) defined in `agents.yaml` must match the strings used in the `config` parameter within `@agent`.
    1. Keys defined in `agents.yaml` can be anything, but should be called carefully while passing to config of `@agent`.

## Agents / Crew Architecture

```
1. TECHNICAL RESEARCHER
    Role: Technical Product Data Specialist
    Goal: Extract specs (RAM, ROM, Processor, Camera, Price) from 3 latest launches
    Output: Technical specifications table
   ↓
   ├─> 2. MARKET STRATEGIST
   │        Role: Consumer Tech Strategy Consultant
   │        Goal: Compare specs, identify value gaps, suggest positioning
   │        Output: Strategic recommendations
   │      ↓
   │      ↓
   └─> 3. PRICE ANALYST
   │        Role: Pricing Intelligence Specialist
   │        Goal: Track price movements, analyze elasticity
   │        Output: Pricing insights & scenarios
   │      ↓
   │      ↓
   └─> 4. SENTIMENT ANALYST
            Role: Consumer Sentiment Analyst
            Goal: Analyze reviews, social media, forums
            Output: Consumer perception analysis
          ↓
5. TREND FORECASTER
    Role: Market Trend Analyst
    Goal: Predict 6-12 month market trends & technology adoption
    Output: Trend forecast
   ↓
6. MARKET SHARE ANALYST
    Role: Market Share Estimator
    Goal: Estimate market share & competitive positioning
    Output: Competitive positioning analysis
   ↓

ALL PATHS CONVERGE → 7. ROI SIMULATOR
                            Role: Business Model Analyst & ROI Strategist
                            Goal: Model financial scenarios and ROI impact
                            Output: Financial validation & scenarios
                     ↓
                     ↓
              FINAL OUTPUTS:
              └─> Market Report
              └─> Trend Forecast

```

## Workflow
Following are top-level steps taken by the pipeline,
1. Create TechMarketCrew() instance. With this initialization, the pipeline now knows location to agents config, tasks config and has access to LLM (currently, gemini).
2. As a user input, pipeline is provided with
    1. category: "Smartphone"
    2. reference_product: "iPhone 15"
3. **Technical Researcher** agent takes in {category, reference_product} as **input** and then uses {DuckDuckGo Search} **tool** to **output** {research_notes.md}.
    1. It searches for 3 latest Smartphones launches in India.
    2. Extract RAM, ROM, Processor, and Camera specs.
    3. Finds retail price ranges on e-commerce platforms.
    4. Formats into markdown table.
4. **Market Strategist** agent takes in {results from Step 3, reference_product} as **input** and then uses its reasoning to **output** {market_report.md}.
    1. It reviews technical specs table from step 3.
    2. Then compare all 3 phones against iPhone 15 {reference_product}.
    3. Perform SWOT analysis for each.
    4. Identify "Value Gaps" where competitors lose to {reference_product}.
    5. Generate 3 actionable recommendations for brand managers.
5. **Trend Forecaster** agent takes in {results from Step 3 and Step 4} as **input** and then uses {DuckDuckGo Search} **tool** to **output** {trend_forecast.md}.
    1. It analyzes current market trends from research data.
    2. Research upcoming smartphone launches.
    3. Identifies feature adoption patterns.
    4. Predicts 6-12 month market trajectory.
    5. Finally, identifies emerging opportunities and threats.
6. **Price Analyst** agent takes in {results from Step 3} as **input** and then uses {DuckDuckGo Search} **tool** to **output** {price_analyst.md}.
    1. Search 6-12 month price history for top 3 phones
    2. Calculate price trends (moving averages, drops, seasonality)
    3. Estimate price elasticity (how demand changes with ±5%, ±10%, ±15% price shifts)
    4. Compare iPhone 15 {reference_product} pricing vs. competitors over time
    5. Identify underpriced and overpriced segments
7. **Market Share Analyst** agent takes in {results from Step 3} as **input** and then uses {DuckDuckGo Search} **tool** to **output** {market_share_analysis.md}.
    1. Search market share data, sales rankings, web traffic
    2. Analyze Google Trends search volume
    3. Track social media mentions and engagement
    4. Estimate sales velocity from e-commerce rankings
    5. Identify gaining vs. losing competitors (6-month trends)
8. **ROI Simulator** agent takes in {results from all steps} as **input** and then uses inherent finanical modelling to **output** {roi_simulation.md}.
    1. Build financial scenarios using price elasticity (Step 4) + sentiment (Step 5)
    2. Model ROI: Price cuts of 5%, 10%, 15% and resulting volume uplift
    3. Calculate margin impact (assume COGS = 60% of price)
    4. Estimate addressable market gains (from Step 6)
    5. Model feature upgrade costs and revenue upside (from Step 2)
    6. Provide breakeven analysis for different strategies

## Information Flow
```
INPUT (category, reference_product)
↓
🔬 Technical Researcher writes → researcher_notes.md
↓
📊 Market Strategist reads ← scratchpad/researcher_notes.md, writes → market_report.md
↓
📈 Trend Forecaster reads ← researcher_notes.md + output/market_report.md, writes → trend_forecast.md
↓
💹 Price Analyst reads ← scratchpad/researcher_notes.md, writes → price_analysis.md
↓
💬 Sentiment Analyst reads ← scratchpad/researcher_notes.md, writes → sentiment_analysis.md
↓
🎯 Market Share Analyst reads ← scratchpad/researcher_notes.md, writes → market_share_analysis.md
↓
💰 ROI Simulator reads ← ALL previous outputs, writes → roi_simulation.md
↓
✅ COMPLETE - All insights ready
```