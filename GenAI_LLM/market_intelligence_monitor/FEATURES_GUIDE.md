# Market Intelligence Monitor - Enhanced Features Guide

## New Features Overview

This document describes the 4 new advanced features added to your Market Intelligence Monitor:

1. **Price History & Elasticity Analysis**
2. **Consumer Sentiment Analysis**
3. **Market Share Estimation**
4. **ROI Simulator**

---

## 1. Price History & Elasticity Analysis

### Purpose
Track historical price trends over 6-12 months and analyze price sensitivity to predict how demand changes with pricing variations.

### What It Does
- **Price History Tracking**: Monitors monthly/quarterly price movements for top products
- **Trend Analysis**: Identifies seasonal patterns and promotional periods
- **Elasticity Estimation**: Calculates price elasticity coefficients for demand sensitivity
- **Scenario Modeling**: Predicts demand changes for ±5%, ±10%, ±15% price variations
- **Segment Analysis**: Identifies price-sensitive (budget) vs. price-insensitive (premium) segments

### Key Outputs
- Historical price trend charts
- Price elasticity coefficients by product/segment
- Volume impact predictions for price changes
- Optimal pricing recommendations

### Business Use Cases
- **Pricing Strategy**: Determine ideal price point to maximize profit (not just revenue)
- **Competitive Response**: React to competitor price cuts with data-driven counter-strategies
- **Segment Optimization**: Use different pricing for budget vs. premium segments
- **Promotional Planning**: Optimal discount levels based on elasticity data

### Example Insight
> "iPhone 15 has an elasticity of -0.8 (inelastic), so a 10% price cut drives only 8% volume increase. However, competitors like Vivo V40 have elasticity of -1.5 (elastic), so a 10% cut drives 15% volume increase. This suggests Vivo should prioritize volume plays while iPhone can maintain premium positioning."

---

## 2. Consumer Sentiment Analysis

### Purpose
Understand what customers actually think about products beyond specs, identifying perception gaps and brand loyalty signals.

### What It Does
- **Multi-Source Sentiment**: Analyzes Amazon reviews, Reddit threads (/r/smartphones, /r/android), tech forums, social media
- **Attribute-Level Sentiment**: Breaks down sentiment by key product attributes:
  - Design & Build Quality
  - Camera Quality
  - Performance/Speed
  - Battery Life
  - Value for Money
- **Perception Gap Analysis**: Identifies where specs outperform perception (underrated) vs. underperform (overrated)
- **Brand Loyalty Metrics**: Measures repeat purchase intent and brand switching signals
- **Sentiment Trends**: Tracks if sentiment is improving or declining over time

### Key Outputs
- Overall sentiment scores (% Positive/Neutral/Negative) by brand
- Attribute-specific sentiment comparison table
- Perception gap identification
- Sentiment trend indicators

### Business Use Cases
- **Marketing Strategy**: Emphasize undervalued features (positive sentiment + low awareness)
- **Product Development**: Address attributes with negative sentiment
- **Brand Positioning**: Differentiate on attributes where you have sentiment advantage
- **Customer Retention**: Identify switching risks from sentiment decline

### Example Insight
> "Samsung Galaxy A54 has excellent camera sentiment (92% positive) but weak value-for-money sentiment (45% positive). Recommendation: Marketing should emphasize camera prowess to justify the premium price. Meanwhile, OnePlus Nord has strong value sentiment but weak camera sentiment—opposite positioning."

---

## 3. Market Share Estimation

### Purpose
Estimate market share using alternative data sources when official data is unavailable.

### What It Does
- **Sales Velocity Tracking**: E-commerce bestseller rankings, inventory levels, restock frequency
- **Search Trend Analysis**: Google Trends for product names, brand comparisons, and buying signals
- **Social Listening**: Brand mentions, hashtags, post engagement, audience growth across platforms
- **Web Traffic Estimation**: Brand site visits, product page traffic, comparison site visibility
- **6-Month Trend Analysis**: Identifies gaining/losing competitors with underlying drivers

### Key Outputs
- Estimated market share (%) by brand
- Market share trend table (6-month view)
- Competitive velocity indicators
- Gaining/losing competitor analysis

### Business Use Cases
- **Competitive Benchmarking**: Track if you're growing or shrinking relative to competitors
- **Early Warning System**: Detect losing ground before official quarterly results
- **Growth Acceleration**: Identify what tactics competitors use for share gains
- **Investment Decisions**: Allocate R&D/marketing resources to where you can capture share

### Example Insight
> "OnePlus market share estimated at 12% (up from 8% six months ago, +50% growth). Growth drivers: aggressive pricing, strong YouTube creator partnerships, and positive review sentiment. iPhone share at 18% (down from 22%), losing switchers to OnePlus value positioning."

---

## 4. ROI Simulator

### Purpose
Build financial "what-if" scenarios to quantify the business impact of strategic recommendations.

### What It Does
- **Price Elasticity Model**: Simulates volume changes for -5%, -10%, -15% price cuts
- **Unit Economics**: Tracks price, COGS, gross margin, revenue, and profit changes
- **Scenario Comparison**: Models 3 different strategic approaches:
  - **Premium Positioning**: Maintain/increase price, target high-value customers
  - **Mid-Market Play**: 10% price cut, target 50-70% of market
  - **Volume Strategy**: 15% price cut, target budget segment
- **Feature ROI**: Model cost and revenue impact of adding new features
- **Breakeven Analysis**: Calculate required volume to maintain current profit levels
- **12-Month Projections**: Annual revenue and profit impact for each scenario

### Key Outputs
- Baseline unit economics (current state)
- Price cut scenarios with profit/margin impact
- Annual revenue and profit projections
- ROI percentage for each scenario
- Recommended strategy with risk assessment

### Business Use Cases
- **Strategic Decision Making**: Quantify the trade-off between margin and volume
- **Investment Justification**: Show profit impact of feature additions (e.g., adding 5G: costs ₹500/unit, drives 20% volume)
- **Feature Prioritization**: Calculate payback period for R&D investments
- **Competitive Response**: Model profitability of aggressive price cuts vs. staying premium

### Example Insight
> "Current state: ₹50,000 price, 1000 units/month = ₹50L revenue, ₹20L profit.
> - **Scenario A (10% cut to ₹45,000)**: 1,150 units (elasticity -1.0) = ₹51.75L revenue, ₹20.7L profit (+3.5% profit ROI)
> - **Scenario B (Add 5G for ₹2,000 cost)**: 1,200 units (+20% volume) = ₹60L revenue, ₹20.4L profit (after cost)
> **Recommendation**: Scenario B better—invest in 5G, drives volume and maintains margins."

---

## How to Use: Integrated Workflow

### Step 1: Run the Full Analysis
The new agents run automatically as part of the sequential workflow. Just run:

```bash
python main.py
```

**Output Flow:**
1. **research_task** → `researcher_notes.md` (product specs)
2. **strategic_analysis_task** → `market_report.md` (SWOT analysis)
3. **trend_forecasting_task** → `trend_forecast.md` (6-12 month outlook)
4. **price_analysis_task** → `price_analysis.md` (elasticity & pricing strategy)
5. **sentiment_analysis_task** → `sentiment_analysis.md` (brand perception)
6. **market_share_task** → `market_share_analysis.md` (competitive position)
7. **roi_simulation_task** → `roi_simulation.md` (financial scenarios)

### Step 2: Review Each Output
All reports are saved in the `output/` directory:
- 7 comprehensive Markdown reports
- Structured with tables, metrics, and actionable recommendations
- Each report includes data sources and confidence levels

### Step 3: Use ROI Simulator Programmatically (Optional)

For deeper analysis, you can use the ROI Simulator utilities directly:

```python
from src.market_crew.utils import ROISimulator, UnitEconomics

# Define baseline unit economics
baseline = UnitEconomics(
    price=50000,              # ₹50,000
    cogs_percentage=0.60,     # 60% of price
    volume_per_month=1000     # Current sales
)

# Create simulator
simulator = ROISimulator(baseline)

# Generate price scenarios
scenarios = simulator.generate_price_scenarios(
    price_changes=[-5, -10, -15],
    elasticity_coefficient=-1.0  # Unit elastic
)

# Print results
for scenario in scenarios:
    print(f"{scenario.scenario_name}: ROI = {scenario.roi_percent:+.1f}%")

# Model feature addition
feature_scenario = simulator.simulate_feature_addition(
    additional_cost_per_unit=2000,
    revenue_uplift_percent=15
)
print(f"Add 5G Feature: {feature_scenario.roi_percent:+.1f}% ROI")
```

---

## Configuration & Customization

### Agents Configuration (`agents.yaml`)

Each new agent has a defined `role`, `goal`, and `backstory` that you can customize:

```yaml
price_analyst:
  role: "Pricing Intelligence Specialist"
  goal: "Track historical price movements for {category} and analyze elasticity"
  backstory: "You are a pricing expert with deep market experience..."
```

To modify behavior:
- **Change the goal**: More specific goals = more focused analysis
- **Adjust backstory**: Different emphasis based on your business context
- **Add constraints**: "Only analyze India market" or "Focus on premium segment"

### Tasks Configuration (`tasks.yaml`)

Each task has:
- `description`: What the agent should do (step-by-step instructions)
- `expected_output`: Format and structure of the report
- `agent`: Which agent executes this task
- `output_file`: Where results are saved

To customize:
- **Modify description**: Add region/segment/category specifics
- **Change output format**: Adjust report structure if needed
- **Update output_file**: Save to different location

### Dependencies

The new features require these packages (already in requirements.txt):
- `crewai`: Multi-agent orchestration
- `langchain-google-genai`: Gemini LLM
- `langchain-community`: DuckDuckGo search tools
- `duckduckgo-search`: Web search capability
- `python-dotenv`: Environment variables

No additional packages needed!

---

## Output Files Reference

| File | Agent | Purpose | Updates Frequency |
|------|-------|---------|-------------------|
| `researcher_notes.md` | Technical Researcher | Product specs & current pricing | Every run |
| `market_report.md` | Market Strategist | Competitive positioning & SWOT | Every run |
| `trend_forecast.md` | Trend Forecaster | 6-12 month market outlook | Every run |
| `price_analysis.md` | Price Analyst | **NEW** - Elasticity & pricing strategy | Every run |
| `sentiment_analysis.md` | Sentiment Analyst | **NEW** - Brand perception & loyalty | Every run |
| `market_share_analysis.md` | Market Share Analyst | **NEW** - Competitive market share | Every run |
| `roi_simulation.md` | ROI Simulator | **NEW** - Financial scenarios & breakeven | Every run |

---

## Performance Tips

1. **Run Overnight**: Full analysis takes 5-15 minutes depending on search results
2. **Cache Results**: Save outputs and only re-run when market conditions change significantly
3. **Customize Per Segment**: Run separately for different market segments (e.g., budget vs. premium)
4. **Use Search Limits**: In tasks.yaml, add "Limit to top 50 search results" to reduce LLM processing time

---

## Troubleshooting

### 1. Sentiment Analysis Getting No Data
- **Issue**: Agent can't find enough reviews/mentions
- **Solution**: Add specific product names to search queries in tasks.yaml
- **Example**: "Find reviews for Samsung Galaxy A54 on Amazon India"

### 2. Price Data Incomplete
- **Issue**: Historical prices not available for all products
- **Solution**: Agent will note data gaps—still provides trend analysis with confidence levels
- **Action**: Consider adding e-commerce price history APIs (e.g., CamelCamelCamel for Amazon)

### 3. Market Share Estimates Seem Off
- **Issue**: No official data for validation
- **Solution**: Use multiple data sources (search trends + social mentions + sales rankings)
- **Action**: Validate against industry reports or sales team estimates

### 4. ROI Scenarios Too Aggressive/Conservative
- **Issue**: Default elasticity coefficient (-1.0) may not match your market
- **Solution**: Adjust elasticity in roi_simulator.py based on historical data
- **Example**: Test with -0.5 (inelastic/premium) or -1.5 (elastic/budget)

---

## Next Steps

### Immediate (Week 1)
1. Run full analysis on your current market
2. Review all 7 outputs
3. Validate sentiment scores against internal feedback
4. Test ROI scenarios against actual historical price changes

### Short Term (Month 1)
1. Set up weekly runs with historical tracking
2. Create dashboard to visualize price trends
3. Integrate market share estimates with sales data
4. Run ROI simulations for upcoming product launches

### Long Term (Ongoing)
1. Build 12-month price elasticity dataset
2. Train custom sentiment models on your product reviews
3. Integrate official market data when available
4. Automate weekly alerts for significant changes

---

## Support & Customization

The framework is fully modular. To extend further:
- **Add new agents**: Define in agents.yaml, create task in tasks.yaml, add method in crew.py
- **Add new tools**: Create tool function, add to agent tools list
- **Modify outputs**: Update expected_output in tasks.yaml
- **Add visualizations**: Create post-processing script using output files

All agents use Google Gemini for reasoning, but you can swap LLM providers by modifying the `llm` initialization in crew.py.

---

**Created**: April 4, 2026  
**Framework**: CrewAI 1.11.0 with Google Gemini 3.1 Flash Lite  
**Status**: Production Ready
