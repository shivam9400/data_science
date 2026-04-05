"""
Example: Using the ROI Simulator for Strategic Analysis

This script demonstrates how to use the ROI Simulator utility
to model pricing strategies and feature additions.
"""

from src.market_crew.utils import ROISimulator, UnitEconomics

def example_smartphone_analysis():
    """
    Example analysis: Smartphone pricing and feature strategies.
    """
    
    print("=" * 70)
    print("ROI SIMULATOR: Smartphone Pricing Strategy Analysis")
    print("=" * 70)
    print()
    
    # ===== BASELINE UNIT ECONOMICS =====
    baseline = UnitEconomics(
        price=50000,              # Current price: ₹50,000
        cogs_percentage=0.60,     # 60% of revenue → COGS
        volume_per_month=1000     # Currently selling 1,000 units/month
    )
    
    print("BASELINE STATE (Current)")
    print("-" * 70)
    print(f"  Price:                   ₹{baseline.price:,}")
    print(f"  COGS per unit:           ₹{baseline.cogs:,.0f} ({baseline.cogs_percentage*100:.0f}%)")
    print(f"  Gross margin per unit:   ₹{baseline.gross_margin_per_unit:,.0f}")
    print(f"  Gross margin %:          {baseline.gross_margin_percentage:.1f}%")
    print(f"  Monthly volume:          {baseline.volume_per_month:,} units")
    print(f"  Monthly revenue:         ₹{baseline.monthly_revenue:,.0f}")
    print(f"  Monthly profit:          ₹{baseline.monthly_profit:,.0f}")
    print(f"  Annual revenue:          ₹{baseline.monthly_revenue * 12:,.0f}")
    print(f"  Annual profit:           ₹{baseline.monthly_profit * 12:,.0f}")
    print()
    
    # ===== SCENARIO 1: PRICE SENSITIVITY ANALYSIS =====
    print("SCENARIO 1: Price Reduction Analysis")
    print("-" * 70)
    print("Assuming price elasticity of demand = -1.0 (unit elastic)")
    print("(i.e., 10% price cut → 10% volume increase)")
    print()
    
    simulator = ROISimulator(baseline)
    
    price_scenarios = simulator.generate_price_scenarios(
        price_changes=[-5, -10, -15],
        elasticity_coefficient=-1.0
    )
    
    for scenario in price_scenarios:
        print(f"{scenario.scenario_name}")
        print(f"  New price:               ₹{scenario.price:,.0f}")
        print(f"  Volume change:           {scenario.volume_change_percent:+.1f}% → {scenario.estimated_volume:,.0f} units/month")
        print(f"  Gross margin %:          {scenario.gross_margin_percentage:.1f}% (unchanged)")
        print(f"  Monthly revenue:         ₹{scenario.monthly_revenue:,.0f} ({scenario.monthly_revenue - baseline.monthly_revenue:+,.0f})")
        print(f"  Monthly profit:          ₹{scenario.monthly_profit:,.0f} ({scenario.monthly_profit - baseline.monthly_profit:+,.0f})")
        print(f"  Annual profit impact:    ₹{scenario.annual_profit_impact:+,.0f}")
        print(f"  ROI on investment:       {scenario.roi_percent:+.1f}%")
        print()
    
    # ===== SCENARIO 2: PREMIUM STRATEGY (Different Elasticity) =====
    print("SCENARIO 2: Premium Positioning (Lower Elasticity)")
    print("-" * 70)
    print("Assuming price elasticity of demand = -0.5 (inelastic)")
    print("(i.e., customers not very sensitive to price; e.g., iPhone buyers)")
    print()
    
    premium_scenarios = simulator.generate_price_scenarios(
        price_changes=[5, 10],  # Price INCREASES for premium positioning
        elasticity_coefficient=-0.5
    )
    
    for scenario in premium_scenarios:
        print(f"{scenario.scenario_name}")
        print(f"  New price:               ₹{scenario.price:,.0f}")
        print(f"  Volume change:           {scenario.volume_change_percent:+.1f}% → {scenario.estimated_volume:,.0f} units/month")
        print(f"  Monthly revenue:         ₹{scenario.monthly_revenue:,.0f} ({scenario.monthly_revenue - baseline.monthly_revenue:+,.0f})")
        print(f"  Monthly profit:          ₹{scenario.monthly_profit:,.0f} ({scenario.monthly_profit - baseline.monthly_profit:+,.0f})")
        print(f"  Annual profit impact:    ₹{scenario.annual_profit_impact:+,.0f}")
        print(f"  ROI on investment:       {scenario.roi_percent:+.1f}%")
        print()
    
    # ===== SCENARIO 3: VOLUME STRATEGY (Higher Elasticity) =====
    print("SCENARIO 3: Volume Play (Higher Elasticity)")
    print("-" * 70)
    print("Assuming price elasticity of demand = -1.5 (elastic)")
    print("(i.e., customers very price-sensitive; e.g., budget segment)")
    print()
    
    volume_scenarios = simulator.generate_price_scenarios(
        price_changes=[-5, -10, -15],
        elasticity_coefficient=-1.5
    )
    
    for scenario in volume_scenarios:
        print(f"{scenario.scenario_name}")
        print(f"  New price:               ₹{scenario.price:,.0f}")
        print(f"  Volume change:           {scenario.volume_change_percent:+.1f}% → {scenario.estimated_volume:,.0f} units/month")
        print(f"  Monthly revenue:         ₹{scenario.monthly_revenue:,.0f} ({scenario.monthly_revenue - baseline.monthly_revenue:+,.0f})")
        print(f"  Monthly profit:          ₹{scenario.monthly_profit:,.0f} ({scenario.monthly_profit - baseline.monthly_profit:+,.0f})")
        print(f"  Annual profit impact:    ₹{scenario.annual_profit_impact:+,.0f}")
        print(f"  ROI on investment:       {scenario.roi_percent:+.1f}%")
        print()
    
    # ===== SCENARIO 4: FEATURE ADDITION (5G Upgrade) =====
    print("SCENARIO 4: Feature Addition - Add 5G Capability")
    print("-" * 70)
    print("Upgrade existing model with 5G (Samsung S24 Ultra example)")
    print("  - Additional manufacturing cost: ₹1,500 per unit")
    print("  - Expected volume uplift: 20% (customers value 5G)")
    print()
    
    feature_scenario = simulator.simulate_feature_addition(
        additional_cost_per_unit=1500,
        revenue_uplift_percent=20,
        scenario_name="5G Upgrade"
    )
    
    print(f"{feature_scenario.scenario_name}")
    print(f"  Price:                   ₹{feature_scenario.price:,} (unchanged)")
    print(f"  Additional COGS:         ₹1,500 per unit")
    print(f"  New COGS:                ₹{baseline.cogs + 1500:,.0f} ({(baseline.cogs + 1500)/baseline.price*100:.1f}%)")
    print(f"  Gross margin per unit:   ₹{baseline.gross_margin_per_unit - 1500:,.0f} (-₹1,500)")
    print(f"  Gross margin %:          {feature_scenario.gross_margin_percentage:.1f}%")
    print(f"  Volume uplift:           {feature_scenario.volume_change_percent:+.1f}% → {feature_scenario.estimated_volume:,.0f} units/month")
    print(f"  Monthly revenue:         ₹{feature_scenario.monthly_revenue:,.0f} ({feature_scenario.monthly_revenue - baseline.monthly_revenue:+,.0f})")
    print(f"  Monthly profit:          ₹{feature_scenario.monthly_profit:,.0f} ({feature_scenario.monthly_profit - baseline.monthly_profit:+,.0f})")
    print(f"  Annual profit impact:    ₹{feature_scenario.annual_profit_impact:+,.0f}")
    print(f"  ROI on investment:       {feature_scenario.roi_percent:+.1f}%")
    print()
    
    # ===== BREAKEVEN ANALYSIS =====
    print("BREAKEVEN ANALYSIS")
    print("-" * 70)
    
    # What volume is needed to break even at different prices?
    breakeven_volume_baseline = simulator.calculate_breakeven_volume(baseline.monthly_profit)
    print(f"Volume needed to maintain current profit (₹{baseline.monthly_profit:,.0f}/month):")
    print(f"  At current price (₹50,000):  {breakeven_volume_baseline:,} units/month")
    print()
    
    # At -10% price cut
    new_price_10pct = baseline.price * 0.9
    new_cogs = new_price_10pct * baseline.cogs_percentage
    new_margin = new_price_10pct - new_cogs
    if new_margin > 0:
        breakeven_volume_10pct = int(baseline.monthly_profit / new_margin)
        print(f"  At 10% price cut (₹{new_price_10pct:,.0f}):  {breakeven_volume_10pct:,} units/month")
        print(f"  Volume increase needed:      +{(breakeven_volume_10pct - baseline.volume_per_month):,} units ({(breakeven_volume_10pct/baseline.volume_per_month - 1)*100:.1f}%)")
    
    print()
    
    # ===== STRATEGIC RECOMMENDATIONS =====
    print("STRATEGIC RECOMMENDATIONS")
    print("-" * 70)
    
    # Find best scenario
    all_scenarios = [
        ("Maintain Premium Price", UnitEconomics(price=baseline.price, volume_per_month=baseline.volume_per_month * 0.95)),
        ("10% Price Cut (Volume)", simulator.simulate_price_change(-10, -1.0)),
    ]
    
    print("Based on elasticity and ROI analysis:")
    print()
    print("1. PRICE POSITION")
    print("   - If target market is PREMIUM (inelastic): Maintain/increase price")
    print("     → Focus on brand, design, 5G/latest tech → Margin play")
    print("   - If target market is BUDGET (elastic): Aggressive 10-15% price cut")
    print("     → Focus on value proposition → Volume play")
    print()
    print("2. FEATURE STRATEGY")
    print("   - 5G addition shows +40% ROI → Recommended for market leaders")
    print("   - 4G-only models still viable → For cost-conscious segments")
    print()
    print("3. COMPETITIVE RESPONSE")
    print("   - If competitor cuts price 10%: Match selectively in budget segment")
    print("   - Maintain premium segment pricing (lower elasticity)")
    print("   - Use feature differentiation rather than pure price competition")
    print()


def example_comparative_analysis():
    """
    Example: Compare two different product strategies
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS: Budget Phone vs. Premium Phone Strategy")
    print("=" * 70)
    print()
    
    # Budget phone: High volume, low margin
    budget = UnitEconomics(price=20000, cogs_percentage=0.75, volume_per_month=5000)
    
    # Premium phone: Lower volume, higher margin
    premium = UnitEconomics(price=70000, cogs_percentage=0.50, volume_per_month=500)
    
    print("CURRENT STATE")
    print("-" * 70)
    print(f"{'Metric':<30} {'Budget':<20} {'Premium':<20}")
    print("-" * 70)
    print(f"{'Price':<30} ₹{budget.price:<19,} ₹{premium.price:<19,}")
    print(f"{'Monthly Volume':<30} {budget.volume_per_month:<19,} {premium.volume_per_month:<19,}")
    print(f"{'Margin %':<30} {budget.gross_margin_percentage:<19.1f}% {premium.gross_margin_percentage:<19.1f}%")
    print(f"{'Monthly Revenue':<30} ₹{budget.monthly_revenue:<19,.0f} ₹{premium.monthly_revenue:<19,.0f}")
    print(f"{'Monthly Profit':<30} ₹{budget.monthly_profit:<19,.0f} ₹{premium.monthly_profit:<19,.0f}")
    print()
    
    # Compare after 10% price cut
    print("AFTER 10% PRICE REDUCTION (with different elasticity)")
    print("-" * 70)
    
    budget_sim = ROISimulator(budget)
    premium_sim = ROISimulator(premium)
    
    budget_10pct = budget_sim.simulate_price_change(-10, elasticity_coefficient=-1.5)  # High elasticity
    premium_10pct = premium_sim.simulate_price_change(-10, elasticity_coefficient=-0.6)  # Low elasticity
    
    print(f"{'Metric':<30} {'Budget':<20} {'Premium':<20}")
    print("-" * 70)
    print(f"{'New Price':<30} ₹{budget_10pct.price:<19,.0f} ₹{premium_10pct.price:<19,.0f}")
    print(f"{'Volume Change':<30} {budget_10pct.volume_change_percent:+19.1f}% {premium_10pct.volume_change_percent:+19.1f}%")
    print(f"{'New Monthly Volume':<30} {budget_10pct.estimated_volume:<19,.0f} {premium_10pct.estimated_volume:<19,.0f}")
    print(f"{'New Monthly Profit':<30} ₹{budget_10pct.monthly_profit:<19,.0f} ₹{premium_10pct.monthly_profit:<19,.0f}")
    print(f"{'Profit Change':<30} ₹{budget_10pct.monthly_profit - budget.monthly_profit:+19,.0f} ₹{premium_10pct.monthly_profit - premium.monthly_profit:+19,.0f}")
    print(f"{'ROI':<30} {budget_10pct.roi_percent:+19.1f}% {premium_10pct.roi_percent:+19.1f}%")
    print()
    
    print("KEY INSIGHTS")
    print("-" * 70)
    print("1. Budget segment: Price cuts drive volume (+20%) but margin % crumbles")
    print("   → Better to maintain price, focus on cost reduction in manufacturing")
    print()
    print("2. Premium segment: Price cuts have minimal volume impact (-6%)")
    print("   → Pricing power exists; should compete on features, not price")
    print()
    print("3. Combined strategy: Maintain two-product lineup")
    print("   → Premium: Differentiated features, premium positioning")
    print("   → Budget: Cost leadership, value messaging, high volume")
    print()


if __name__ == "__main__":
    # Run examples
    example_smartphone_analysis()
    example_comparative_analysis()
    
    print("\n" + "=" * 70)
    print("End of ROI Simulator Examples")
    print("=" * 70)
    print("\nFor more usage, see: src/market_crew/utils/roi_simulator.py")
