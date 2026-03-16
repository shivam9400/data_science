"""
Overtraining Detection Script - Analyzes Strava running data for overtraining indicators
Based on heart rate trends, pace degradation, and recovery metrics
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


class OvertrainingAnalyzer:
    """Analyzes running activities for overtraining indicators"""
    
    # Constants for overtraining detection
    RESTING_HR_NORMAL = 60  # Your estimated resting HR
    MAX_HR = 192  # Your estimated max HR (based on Comeback Run)
    HR_RESERVE = MAX_HR - RESTING_HR_NORMAL
    
    # Overtraining thresholds
    ZONE_5_THRESHOLD = 0.90 * MAX_HR  # 90% of max HR = Zone 5 (anaerobic)
    SUSTAINED_HIGH_HR_THRESHOLD = 0.85 * MAX_HR  # 85% of max HR
    HR_VOLATILITY_WARNING = 15  # bpm increase between runs
    PACE_DEGRADATION_WARNING = 0.5  # km/h slower
    
    def __init__(self, activities_json_path=None):
        """Initialize with optional JSON data path"""
        self.activities = []
        self.last_3_runs = []
        
        if activities_json_path and Path(activities_json_path).exists():
            self.load_from_file(activities_json_path)
    
    def load_from_file(self, json_path):
        """Load activities from Strava JSON export"""
        try:
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            self.activities = data.get('activities', [])
            self._extract_last_3_runs()
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def _extract_last_3_runs(self):
        """Extract last 3 actual running activities"""
        runs = [a for a in self.activities if a.get('type') == 'Run']
        self.last_3_runs = sorted(runs, key=lambda x: x['start_date'], reverse=True)[:3]
    
    def get_run_metrics(self, run):
        """Extract key metrics from a single run"""
        distance_km = run.get('distance', 0) / 1000
        time_seconds = run.get('moving_time', 0)
        time_hours = time_seconds / 3600
        
        # Calculate pace (min/km)
        if distance_km > 0:
            pace_min_per_km = (time_seconds / 60) / distance_km
            pace_kmh = distance_km / time_hours if time_hours > 0 else 0
        else:
            pace_min_per_km = 0
            pace_kmh = 0
        
        avg_hr = run.get('average_heartrate')
        max_hr = run.get('max_heartrate')
        
        return {
            'date': run.get('start_date_local', '')[:10],
            'name': run.get('name', 'Run'),
            'distance_km': distance_km,
            'time_hours': time_hours,
            'pace_min_per_km': pace_min_per_km,
            'pace_kmh': pace_kmh,
            'avg_hr': avg_hr,
            'max_hr': max_hr,
            'moving_time_sec': time_seconds
        }
    
    def calculate_hr_zones(self, avg_hr):
        """Calculate which Heart Rate Zone the activity is in"""
        if avg_hr is None:
            return "Unknown"
        
        hr_percent = (avg_hr / self.MAX_HR) * 100
        
        if hr_percent < 60:
            return "Zone 1 (Recovery)"
        elif hr_percent < 70:
            return "Zone 2 (Endurance)"
        elif hr_percent < 80:
            return "Zone 3 (Tempo)"
        elif hr_percent < 90:
            return "Zone 4 (Threshold)"
        else:
            return "Zone 5 (VO2 Max/Anaerobic)"
    
    def detect_overtraining_indicators(self):
        """Analyze last 3 runs for overtraining indicators"""
        if len(self.last_3_runs) < 1:
            return {
                'status': 'INSUFFICIENT_DATA',
                'message': 'Not enough run data to analyze',
                'indicators': []
            }
        
        metrics = [self.get_run_metrics(run) for run in self.last_3_runs]
        indicators = []
        risk_level = 'LOW'
        
        print("\n" + "="*80)
        print("OVERTRAINING ANALYSIS REPORT")
        print("="*80)
        
        # Display run metrics
        print("\nLAST 3 RUNS ANALYSIS:")
        print("-"*80)
        for i, m in enumerate(metrics, 1):
            print(f"\nRun {i}: {m['date']} - {m['name']}")
            print(f"  Distance: {m['distance_km']:.2f} km")
            print(f"  Duration: {m['time_hours']:.2f} hours ({int(m['moving_time_sec']/60)} min)")
            print(f"  Pace: {m['pace_min_per_km']:.2f} min/km ({m['pace_kmh']:.2f} km/h)")
            print(f"  Avg HR: {m['avg_hr']} bpm" if m['avg_hr'] else "  Avg HR: No data")
            print(f"  Max HR: {m['max_hr']} bpm" if m['max_hr'] else "  Max HR: No data")
            if m['avg_hr']:
                print(f"  Zone: {self.calculate_hr_zones(m['avg_hr'])}")
        
        # Check 1: Sustained High Heart Rate (Indicator of stress)
        print("\n" + "-"*80)
        print("OVERTRAINING INDICATORS:")
        print("-"*80)
        
        high_hr_runs = [m for m in metrics if m['avg_hr'] and m['avg_hr'] > self.SUSTAINED_HIGH_HR_THRESHOLD]
        if len(high_hr_runs) >= 2:
            indicators.append({
                'type': 'HIGH_SUSTAINED_HR',
                'severity': 'MODERATE',
                'description': f'{len(high_hr_runs)} of last 3 runs sustained HR > {self.SUSTAINED_HIGH_HR_THRESHOLD} bpm (Zone 4-5)',
                'runs_affected': [m['date'] for m in high_hr_runs]
            })
            risk_level = 'MODERATE'
            print(f"\n⚠️  ELEVATED HR ZONES: {len(high_hr_runs)} runs in high intensity zones")
            for m in high_hr_runs:
                print(f"    {m['date']}: {m['avg_hr']} bpm ({(m['avg_hr']/self.MAX_HR)*100:.1f}% max)")
        else:
            print("\n✓ Heart rate zones appear normal")
        
        # Check 2: HR Volatility (Sign of incomplete recovery)
        if len(metrics) >= 2:
            hr_changes = []
            for i in range(len(metrics)-1):
                if metrics[i]['avg_hr'] and metrics[i+1]['avg_hr']:
                    change = metrics[i+1]['avg_hr'] - metrics[i]['avg_hr']
                    hr_changes.append(change)
                    
                    if abs(change) > self.HR_VOLATILITY_WARNING:
                        indicators.append({
                            'type': 'HR_VOLATILITY',
                            'severity': 'MODERATE',
                            'description': f'HR increased {abs(change):.1f} bpm between runs (Run {i+1} to {i+2})',
                            'change': change
                        })
                        print(f"\n⚠️  HR VOLATILITY DETECTED")
                        print(f"    {metrics[i+1]['date']}: HR increased {change:.1f} bpm from previous run")
                        print(f"    This can indicate incomplete recovery between hard efforts")
                        risk_level = 'MODERATE'
            
            if hr_changes and all(abs(c) <= self.HR_VOLATILITY_WARNING for c in hr_changes):
                print("\n✓ Heart rate recovery appears stable")
        
        # Check 3: Pace Degradation (Sign of fatigue)
        if len(metrics) >= 2:
            pace_degradation = []
            for i in range(len(metrics)-1):
                pace_diff = metrics[i]['pace_kmh'] - metrics[i+1]['pace_kmh']
                if pace_diff > self.PACE_DEGRADATION_WARNING:
                    pace_degradation.append(pace_diff)
                    indicators.append({
                        'type': 'PACE_DEGRADATION',
                        'severity': 'MODERATE',
                        'description': f'Pace dropped {pace_diff:.2f} km/h between runs',
                        'pace_drop': pace_diff
                    })
            
            if pace_degradation:
                print(f"\n⚠️  PACE DEGRADATION DETECTED")
                for i, deg in enumerate(pace_degradation):
                    print(f"    Run {i+1} to {i+2}: {deg:.2f} km/h slower")
                    print(f"    Could indicate neuromuscular fatigue")
                risk_level = 'MODERATE'
            else:
                print("\n✓ Pace relatively consistent")
        
        # Check 4: Recovery Time Between Runs
        if len(self.last_3_runs) >= 2:
            print(f"\n" + "-"*80)
            print("RECOVERY TIME ANALYSIS:")
            print("-"*80)
            
            recovery_times = []
            for i in range(len(self.last_3_runs)-1):
                date1 = datetime.fromisoformat(self.last_3_runs[i]['start_date'].replace('Z', '+00:00'))
                date2 = datetime.fromisoformat(self.last_3_runs[i+1]['start_date'].replace('Z', '+00:00'))
                hours_between = (date1 - date2).total_seconds() / 3600
                recovery_times.append(hours_between)
                
                print(f"\nRun {i+1} to Run {i+2}:")
                print(f"  {self.last_3_runs[i+1]['start_date_local'][:10]} -> {self.last_3_runs[i]['start_date_local'][:10]}")
                print(f"  Recovery time: {hours_between:.1f} hours ({hours_between/24:.1f} days)")
                
                if hours_between < 24:
                    indicators.append({
                        'type': 'INSUFFICIENT_RECOVERY',
                        'severity': 'HIGH',
                        'description': f'Only {hours_between:.1f} hours recovery between hard efforts',
                        'hours': hours_between
                    })
                    print(f"  ⚠️  Less than 24 hours - may be insufficient recovery!")
                    risk_level = 'HIGH'
                elif hours_between < 48 and metrics[i]['avg_hr'] and metrics[i]['avg_hr'] > 160:
                    print(f"  ⚠️  Back-to-back high intensity runs - monitor carefully")
                else:
                    print(f"  ✓ Good recovery window")
        
        # Overall Summary
        print(f"\n" + "="*80)
        print("OVERTRAINING RISK ASSESSMENT")
        print("="*80)
        
        if risk_level == 'HIGH':
            print(f"\n🔴 RISK LEVEL: HIGH")
            print("\nRECOMMENDATION: INCREASE RECOVERY")
            print("  • Add 1-2 extra rest days")
            print("  • Take easy 20-30 min runs only for next 3-4 days")
            print("  • Prioritize sleep and nutrition")
            print("  • Consider skipping strength training until recovered")
        elif risk_level == 'MODERATE':
            print(f"\n🟡 RISK LEVEL: MODERATE")
            print("\nRECOMMENDATION: MONITOR CLOSELY")
            print("  • Ensure 48+ hours between hard efforts")
            print("  • Include at least 2 easy/recovery runs per week")
            print("  • Track resting heart rate - increase of >5 bpm indicates overtraining")
            print("  • Consider reducing intensity of next 1-2 workouts")
        else:
            print(f"\n🟢 RISK LEVEL: LOW")
            print("\nRECOMMENDATION: CONTINUE CURRENT TRAINING")
            print("  • Recovery patterns look good")
            print("  • Continue following your planned training schedule")
            print("  • Monitor HR and pace trends weekly")
        
        print("\n" + "="*80)
        print(f"Total indicators found: {len(indicators)}")
        print("="*80 + "\n")
        
        return {
            'status': 'ANALYZED',
            'risk_level': risk_level,
            'indicators': indicators,
            'metrics': metrics
        }


def main():
    """Main execution"""
    import sys
    
    # Try to load from Strava data
    json_path = r"c:\Users\Shivam Sharma\AppData\Roaming\Code\User\workspaceStorage\2eebaa3e4223e56b57a55defdcb5cf5d\GitHub.copilot-chat\chat-session-resources\3c08277e-eebe-4cd4-8c22-7a9ba79ab42b\toolu_vrtx_01F1UvWSoB7ydQWdzDWzbfLW__vscode-1773595814700\content.json"
    
    analyzer = OvertrainingAnalyzer(json_path)
    
    if not analyzer.last_3_runs:
        print("Error: Could not load Strava data. Make sure the file path is correct.")
        sys.exit(1)
    
    result = analyzer.detect_overtraining_indicators()
    
    return result


if __name__ == "__main__":
    main()
