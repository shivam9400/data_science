"""
ROI Simulator Utility
Provides functions for financial scenario modeling and ROI calculations.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class UnitEconomics:
    """Represents unit economics for a product."""
    price: float  # Price in INR
    cogs_percentage: float = 0.60  # Cost of Goods Sold as % of price (default 60%)
    volume_per_month: float = 1000  # Current monthly volume
    
    @property
    def cogs(self) -> float:
        """Calculate COGS per unit."""
        return self.price * self.cogs_percentage
    
    @property
    def gross_margin_per_unit(self) -> float:
        """Calculate gross margin per unit."""
        return self.price - self.cogs
    
    @property
    def gross_margin_percentage(self) -> float:
        """Calculate gross margin percentage."""
        return (self.gross_margin_per_unit / self.price) * 100
    
    @property
    def monthly_revenue(self) -> float:
        """Calculate current monthly revenue."""
        return self.price * self.volume_per_month
    
    @property
    def monthly_profit(self) -> float:
        """Calculate current monthly profit."""
        return self.gross_margin_per_unit * self.volume_per_month


@dataclass
class ScenarioResult:
    """Results from a pricing scenario."""
    scenario_name: str
    price: float
    price_change_percent: float
    estimated_volume: float
    volume_change_percent: float
    monthly_revenue: float
    monthly_profit: float
    gross_margin_percentage: float
    annual_revenue_impact: float
    annual_profit_impact: float
    roi_percent: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting."""
        return {
            'scenario_name': self.scenario_name,
            'price': f"₹{self.price:,.0f}",
            'price_change': f"{self.price_change_percent:+.1f}%",
            'volume': f"{self.estimated_volume:,.0f} units/month",
            'volume_change': f"{self.volume_change_percent:+.1f}%",
            'monthly_revenue': f"₹{self.monthly_revenue:,.0f}",
            'monthly_profit': f"₹{self.monthly_profit:,.0f}",
            'margin': f"{self.gross_margin_percentage:.1f}%",
            'annual_revenue': f"₹{self.annual_revenue_impact:,.0f}",
            'annual_profit': f"₹{self.annual_profit_impact:,.0f}",
            'roi': f"{self.roi_percent:+.1f}%"
        }


class PricingElasticity:
    """Model price elasticity and demand sensitivity."""
    
    @staticmethod
    def estimate_volume_change(elasticity_coefficient: float, price_change_percent: float) -> float:
        """
        Calculate volume change based on price elasticity.
        
        Formula: % Change in Quantity Demanded = Elasticity × % Change in Price
        
        Args:
            elasticity_coefficient: Price elasticity of demand (typically -0.5 to -2.0)
            price_change_percent: Percentage change in price (e.g., -10 for 10% decrease)
        
        Returns:
            Percentage change in volume demanded
        """
        return elasticity_coefficient * price_change_percent


class ROISimulator:
    """Main ROI simulation engine."""
    
    def __init__(self, baseline: UnitEconomics):
        """
        Initialize with baseline unit economics.
        
        Args:
            baseline: UnitEconomics object with current state
        """
        self.baseline = baseline
    
    def simulate_price_change(
        self, 
        price_change_percent: float, 
        elasticity_coefficient: float = -1.0,
        scenario_name: str = None
    ) -> ScenarioResult:
        """
        Simulate a price change scenario.
        
        Args:
            price_change_percent: Price change as percentage (e.g., -10 for 10% cut)
            elasticity_coefficient: Price elasticity coefficient (default -1.0 = unit elastic)
            scenario_name: Optional name for this scenario
        
        Returns:
            ScenarioResult with financial projections
        """
        if scenario_name is None:
            scenario_name = f"Price {price_change_percent:+.0f}%"
        
        # Calculate new price
        new_price = self.baseline.price * (1 + price_change_percent / 100)
        
        # Estimate volume change using elasticity
        estimated_volume_change = PricingElasticity.estimate_volume_change(
            elasticity_coefficient, 
            price_change_percent
        )
        new_volume = self.baseline.volume_per_month * (1 + estimated_volume_change / 100)
        
        # Calculate new economics
        new_cogs = new_price * self.baseline.cogs_percentage
        new_margin_per_unit = new_price - new_cogs
        new_gross_margin_pct = (new_margin_per_unit / new_price) * 100
        new_monthly_revenue = new_price * new_volume
        new_monthly_profit = new_margin_per_unit * new_volume
        
        # Calculate impacts
        annual_revenue_impact = (new_monthly_revenue - self.baseline.monthly_revenue) * 12
        annual_profit_impact = (new_monthly_profit - self.baseline.monthly_profit) * 12
        
        # Calculate ROI on investment (assuming any price cut is an "investment" in volume)
        if self.baseline.monthly_profit > 0:
            roi_percent = (annual_profit_impact / (self.baseline.monthly_profit * 12)) * 100
        else:
            roi_percent = 0.0
        
        return ScenarioResult(
            scenario_name=scenario_name,
            price=new_price,
            price_change_percent=price_change_percent,
            estimated_volume=new_volume,
            volume_change_percent=estimated_volume_change,
            monthly_revenue=new_monthly_revenue,
            monthly_profit=new_monthly_profit,
            gross_margin_percentage=new_gross_margin_pct,
            annual_revenue_impact=annual_revenue_impact,
            annual_profit_impact=annual_profit_impact,
            roi_percent=roi_percent
        )
    
    def generate_price_scenarios(
        self, 
        price_changes: List[float] = None,
        elasticity_coefficient: float = -1.0
    ) -> List[ScenarioResult]:
        """
        Generate multiple pricing scenarios.
        
        Args:
            price_changes: List of price change percentages (e.g., [-5, -10, -15])
            elasticity_coefficient: Price elasticity coefficient
        
        Returns:
            List of ScenarioResult objects
        """
        if price_changes is None:
            price_changes = [-5, -10, -15]
        
        results = []
        for price_change in price_changes:
            result = self.simulate_price_change(
                price_change, 
                elasticity_coefficient
            )
            results.append(result)
        
        return results
    
    def calculate_breakeven_volume(self, target_monthly_profit: float) -> int:
        """
        Calculate volume needed to achieve target monthly profit.
        
        Args:
            target_monthly_profit: Target profit in INR per month
        
        Returns:
            Required volume (units per month)
        """
        if self.baseline.gross_margin_per_unit == 0:
            return 0
        
        return int(target_monthly_profit / self.baseline.gross_margin_per_unit)
    
    def simulate_feature_addition(
        self,
        additional_cost_per_unit: float,
        revenue_uplift_percent: float,
        scenario_name: str = "Feature Addition"
    ) -> ScenarioResult:
        """
        Simulate adding a feature to the product.
        
        Args:
            additional_cost_per_unit: Added manufacturing cost in INR
            revenue_uplift_percent: Expected revenue uplift percentage (e.g., 15 for 15% higher volume)
            scenario_name: Name for this scenario
        
        Returns:
            ScenarioResult of the feature addition scenario
        """
        new_cogs = self.baseline.cogs + additional_cost_per_unit
        new_margin_per_unit = self.baseline.price - new_cogs
        new_volume = self.baseline.volume_per_month * (1 + revenue_uplift_percent / 100)
        new_monthly_revenue = self.baseline.price * new_volume
        new_monthly_profit = new_margin_per_unit * new_volume
        new_gross_margin_pct = (new_margin_per_unit / self.baseline.price) * 100
        
        annual_revenue_impact = (new_monthly_revenue - self.baseline.monthly_revenue) * 12
        annual_profit_impact = (new_monthly_profit - self.baseline.monthly_profit) * 12
        roi_percent = (annual_profit_impact / (self.baseline.monthly_profit * 12)) * 100 if self.baseline.monthly_profit > 0 else 0
        
        return ScenarioResult(
            scenario_name=scenario_name,
            price=self.baseline.price,
            price_change_percent=0,
            estimated_volume=new_volume,
            volume_change_percent=revenue_uplift_percent,
            monthly_revenue=new_monthly_revenue,
            monthly_profit=new_monthly_profit,
            gross_margin_percentage=new_gross_margin_pct,
            annual_revenue_impact=annual_revenue_impact,
            annual_profit_impact=annual_profit_impact,
            roi_percent=roi_percent
        )
    
    def payback_period_months(self, feature_cost_per_unit: float, volume_monthly: float) -> float:
        """
        Calculate payback period for a feature investment.
        
        Args:
            feature_cost_per_unit: Cost of feature per unit
            volume_monthly: Expected monthly sales volume
        
        Returns:
            Payback period in months
        """
        total_feature_cost = feature_cost_per_unit * volume_monthly
        margin_per_unit = self.baseline.gross_margin_per_unit + feature_cost_per_unit  # Assume margin maintenance
        monthly_margin_from_feature = margin_per_unit * volume_monthly - self.baseline.monthly_profit
        
        if monthly_margin_from_feature <= 0:
            return float('inf')
        
        return total_feature_cost / monthly_margin_from_feature


# Example usage function
def example_roi_analysis():
    """Example of how to use the ROI Simulator."""
    # Define baseline economics
    baseline = UnitEconomics(
        price=50000,  # ₹50,000
        cogs_percentage=0.60,
        volume_per_month=1000
    )
    
    # Create simulator
    simulator = ROISimulator(baseline)
    
    # Generate scenarios
    scenarios = simulator.generate_price_scenarios(
        price_changes=[-5, -10, -15],
        elasticity_coefficient=-1.0
    )
    
    print("=== ROI Analysis: Price Sensitivity ===")
    print(f"Baseline: ₹{baseline.price:,} | Volume: {baseline.volume_per_month:,} units/month")
    print(f"Monthly Revenue: ₹{baseline.monthly_revenue:,.0f} | Profit: ₹{baseline.monthly_profit:,.0f}")
    print()
    
    for scenario in scenarios:
        print(f"{scenario.scenario_name}:")
        for key, value in scenario.to_dict().items():
            print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    example_roi_analysis()
