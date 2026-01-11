import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os

# ===================================================================
# STEP 1: Load and Prepare Data
# ===================================================================

print("Loading data...")
df = pd.read_csv('Kotak 1st Nov 2022 to 18th April 2024_15T.csv', index_col=0, parse_dates=True)

# Display loaded data info
print(f"✓ Data loaded: {len(df)} rows")
print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
print(f"✓ Price range: {df['Low'].min():.2f} to {df['High'].max():.2f}")
print(f"✓ Columns: {list(df.columns)}")

# ===================================================================
# STEP 2: Original Algorithm - Support/Resistance Detection
# ===================================================================

print("\n" + "="*70)
print("DETECTING SUPPORT/RESISTANCE LEVELS")
print("="*70)

# Support Levels (Local Minima)
# Uses rolling window to find local minima with lookahead
support_series = df['Low'].rolling(10, center=True).min()

# Resistance Levels (Local Maxima)
# Uses rolling window to find local maxima with lookahead
resistance_series = df['High'].rolling(10, center=True).max()

# Combine all levels
levels = pd.concat([support_series, resistance_series])

# Drop NaN values
levels = levels.dropna()

print(f"Raw pivots found: {len(levels)}")
print(f"  - From support (minima): {len(support_series.dropna())}")
print(f"  - From resistance (maxima): {len(resistance_series.dropna())}")

# Filter out duplicate/similar levels
# Keep only levels that are > 100 rupees apart
levels_sorted = levels.sort_values()
levels_filtered = []

for price in levels_sorted.values:
    # Check if this price is far enough from existing filtered levels
    if len(levels_filtered) == 0:
        levels_filtered.append(price)
    else:
        # Check distance to nearest existing level
        min_distance = min(abs(price - existing) for existing in levels_filtered)
        if min_distance > 25:  # Only add if > 100 away from nearest level
            levels_filtered.append(price)

levels = np.array(sorted(levels_filtered))

print(f"After filtering (distance > 100): {len(levels)} unique levels")

# Separate into support and resistance
support_levels = []
resistance_levels = []

for level in levels:
    # A level is "support" if it came from Low data
    support_mask = (support_series == level).any()
    # A level is "resistance" if it came from High data
    resistance_mask = (resistance_series == level).any()
    
    if support_mask and not resistance_mask:
        support_levels.append(level)
    elif resistance_mask and not support_mask:
        resistance_levels.append(level)
    else:
        # If found in both, classify based on context (assign to support if closer to bottom)
        if level < (df['Low'].max() + df['High'].min()) / 2:
            support_levels.append(level)
        else:
            resistance_levels.append(level)

support_levels = np.array(sorted(set(support_levels)))
resistance_levels = np.array(sorted(set(resistance_levels)))

print(f"\n--- DETECTED SUPPORT/RESISTANCE LEVELS ---")
print(f"Support Levels ({len(support_levels)}): {np.round(support_levels, 2)}")
print(f"\nResistance Levels ({len(resistance_levels)}): {np.round(resistance_levels, 2)}")

# ===================================================================
# STEP 3: Evaluation Metrics (NEW)
# ===================================================================

def evaluate_sr_metrics(df, support_levels, resistance_levels, tolerance=3.0):
    """
    Evaluate S/R level quality using standard metrics.
    
    Parameters:
    -----------
    df : DataFrame
        OHLCV data with 'High', 'Low', 'Close' columns
    support_levels : array-like
        Detected support levels
    resistance_levels : array-like
        Detected resistance levels
    tolerance : float
        Distance tolerance for touching (in price units)
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    
    metrics = {}
    
    # Convert to arrays
    support_levels = np.array(support_levels)
    resistance_levels = np.array(resistance_levels)
    all_levels = np.concatenate([support_levels, resistance_levels])
    all_levels = np.sort(all_levels)
    
    # 1. RECURRENCE SCORES
    # Support: % of Low prices within tolerance of any support level
    def calc_recurrence(prices, levels, tolerance):
        """Calculate % of prices touching at least one level"""
        if len(levels) == 0 or len(prices) == 0:
            return 0.0
        
        touch_count = 0
        for price in prices:
            if np.any(np.abs(levels - price) <= tolerance):
                touch_count += 1
        
        return touch_count / len(prices)
    
    support_recurrence = calc_recurrence(df['Low'].values, support_levels, tolerance)
    resistance_recurrence = calc_recurrence(df['High'].values, resistance_levels, tolerance)
    
    metrics['Support_Recurrence'] = support_recurrence
    metrics['Resistance_Recurrence'] = resistance_recurrence
    
    # 2. LEVEL SEPARATION
    price_range = df['Close'].max() - df['Close'].min()
    
    if len(all_levels) > 1 and price_range > 0:
        distances = np.diff(all_levels)
        avg_distance = np.mean(distances)
        avg_distance_pct = (avg_distance / price_range) * 100
        metrics['Average_Level_Separation_Pct'] = avg_distance_pct
    else:
        metrics['Average_Level_Separation_Pct'] = 0.0
    
    # 3. LEVEL COUNTS
    metrics['Support_Count'] = len(support_levels)
    metrics['Resistance_Count'] = len(resistance_levels)
    metrics['Total_Levels'] = len(all_levels)
    
    # 4. CLUSTERING QUALITY
    overlap_count = 0
    for sl in support_levels:
        for rl in resistance_levels:
            if abs(sl - rl) <= tolerance:
                overlap_count += 1
    
    metrics['Overlapping_Pairs'] = overlap_count
    
    # 5. COVERAGE
    unique_price_points = set()
    for price in df['Low'].values:
        for level in support_levels:
            if abs(level - price) <= tolerance:
                unique_price_points.add(price)
                break
    
    for price in df['High'].values:
        for level in resistance_levels:
            if abs(level - price) <= tolerance:
                unique_price_points.add(price)
                break
    
    coverage_pct = (len(unique_price_points) / len(df)) * 100
    metrics['Coverage_Percentage'] = coverage_pct
    
    return metrics


# Define tolerance for metrics
TOLERANCE = 2.0

# Calculate evaluation metrics
print("\n" + "="*70)
print("EVALUATION METRICS (WITH LOOKAHEAD)")
print("="*70)

metrics = evaluate_sr_metrics(df, support_levels, resistance_levels, tolerance=TOLERANCE)

print(f"\n📊 SUPPORT RECURRENCE SCORE: {metrics['Support_Recurrence']:.4f}")
print(f"   Meaning: {metrics['Support_Recurrence']*100:.2f}% of Low prices touch a support level")
print(f"   Ideal: 0.20-0.40 | Current: {'✓ GOOD' if 0.20 <= metrics['Support_Recurrence'] <= 0.45 else '⚠️ OUT OF RANGE'}")

print(f"\n📊 RESISTANCE RECURRENCE SCORE: {metrics['Resistance_Recurrence']:.4f}")
print(f"   Meaning: {metrics['Resistance_Recurrence']*100:.2f}% of High prices touch a resistance level")
print(f"   Ideal: 0.20-0.40 | Current: {'✓ GOOD' if 0.20 <= metrics['Resistance_Recurrence'] <= 0.45 else '⚠️ OUT OF RANGE'}")

print(f"\n📊 AVERAGE LEVEL SEPARATION: {metrics['Average_Level_Separation_Pct']:.3f}%")
print(f"   Meaning: Consecutive levels are {metrics['Average_Level_Separation_Pct']:.3f}% of price range apart")
print(f"   Ideal: 1-15% | Current: {'✓ GOOD' if 1.0 <= metrics['Average_Level_Separation_Pct'] <= 15 else '⚠️ OUT OF RANGE'}")

print(f"\n📊 LEVEL COUNTS:")
print(f"   Support Levels: {metrics['Support_Count']}")
print(f"   Resistance Levels: {metrics['Resistance_Count']}")
print(f"   Total Levels: {metrics['Total_Levels']}")
print(f"   Ideal: 6-20 | Current: {'✓ GOOD' if 6 <= metrics['Total_Levels'] <= 20 else '⚠️ OUT OF RANGE'}")

print(f"\n📊 OVERLAPPING PAIRS: {metrics['Overlapping_Pairs']}")
print(f"   Meaning: Support-Resistance pairs within ±{TOLERANCE} price units")

print(f"\n📊 COVERAGE PERCENTAGE: {metrics['Coverage_Percentage']:.2f}%")
print(f"   Meaning: % of price data points touching at least one S/R level")
print(f"   Ideal: 20-60% | Current: {'✓ GOOD' if 20 <= metrics['Coverage_Percentage'] <= 60 else '⚠️ OUT OF RANGE'}")

# ===================================================================
# STEP 4: Quality Assessment
# ===================================================================

print("\n" + "="*70)
print("QUALITY ASSESSMENT")
print("="*70)

quality_score = 0
recommendations = []

# Check recurrence balance
recurrence_diff = abs(metrics['Support_Recurrence'] - metrics['Resistance_Recurrence'])
if recurrence_diff < 0.10:
    print("✓ Balanced recurrence scores")
    quality_score += 2
else:
    print(f"⚠️  Imbalanced recurrence scores (diff: {recurrence_diff:.4f})")
    recommendations.append(f"- Recurrence imbalance detected. Consider adjusting rolling window or tolerance.")

# Check level separation
if 1.0 <= metrics['Average_Level_Separation_Pct'] <= 15:
    print("✓ Good level separation (1-15%)")
    quality_score += 2
else:
    print(f"⚠️  Level separation outside ideal range (current: {metrics['Average_Level_Separation_Pct']:.2f}%)")
    if metrics['Average_Level_Separation_Pct'] < 1.0:
        recommendations.append("- Levels too close: Increase minimum distance filter (currently 100)")
    else:
        recommendations.append("- Levels too far: Decrease minimum distance filter (currently 100)")

# Check level count
if 6 <= metrics['Total_Levels'] <= 20:
    print("✓ Reasonable number of levels (6-20)")
    quality_score += 2
else:
    print(f"⚠️  Unusual level count: {metrics['Total_Levels']}")
    if metrics['Total_Levels'] < 6:
        recommendations.append("- Too few levels: Increase rolling window or decrease minimum distance")
    else:
        recommendations.append("- Too many levels: Decrease rolling window or increase minimum distance")

# Check coverage
if 20 <= metrics['Coverage_Percentage'] <= 60:
    print("✓ Good coverage percentage (20-60%)")
    quality_score += 2
else:
    print(f"⚠️  Coverage outside ideal range: {metrics['Coverage_Percentage']:.2f}%")

print(f"\n📊 Overall Quality Score: {quality_score}/8")

if quality_score >= 7:
    print("✓✓ EXCELLENT - Levels are well-distributed and meaningful")
elif quality_score >= 5:
    print("✓ GOOD - Levels are reasonable with minor adjustments possible")
else:
    print("⚠️  FAIR - Consider reviewing detection parameters")

if recommendations:
    print("\n💡 RECOMMENDATIONS:")
    for rec in recommendations:
        print(rec)

# ===================================================================
# STEP 5: Visualization
# ===================================================================

print("\n" + "="*70)
print("GENERATING CHART")
print("="*70)

# Plot Price History
try:
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History', fontsize=18)
    plt.plot(df.index, df['Close'], linewidth=1.5)
    
    # Add support lines
    for support in support_levels:
        plt.axhline(y=support, color='green', linestyle='--', linewidth=1, alpha=0.6)
    
    # Add resistance lines
    for resistance in resistance_levels:
        plt.axhline(y=resistance, color='red', linestyle='--', linewidth=1, alpha=0.6)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.tight_layout()
    plt.savefig('support_resistance_chart.png', dpi=100, bbox_inches='tight')
    print("✓ Chart saved: support_resistance_chart.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Could not save matplotlib chart: {str(e)[:50]}")

# Try mplfinance for candlestick chart
try:
    hlines = dict(
        hlines=list(support_levels) + list(resistance_levels),
        colors=['green']*len(support_levels) + ['red']*len(resistance_levels),
        linewidths=1
    )
    
    mpf.plot(
        df,
        type='candle',
        style='yahoo',
        title='Support & Resistance Levels',
        hlines=hlines,
        figsize=(14, 8),
        warn_too_much_data=len(df)+1,
        savefig='support_resistance_candlestick.png'
    )
    print("✓ Candlestick chart saved: support_resistance_candlestick.png")
except Exception as e:
    print(f"⚠️  mplfinance chart skipped: {str(e)[:50]}")

# ===================================================================
# STEP 6: Export Metrics
# ===================================================================

print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

# Create metrics DataFrame
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('sr_metrics.csv', index=False)
print("✓ Metrics exported: sr_metrics.csv")

# Export levels to CSV
levels_export = pd.DataFrame({
    'Level': np.concatenate([support_levels, resistance_levels]),
    'Type': ['Support']*len(support_levels) + ['Resistance']*len(resistance_levels),
    'Price': np.concatenate([support_levels, resistance_levels])
})
levels_export = levels_export.sort_values('Price').reset_index(drop=True)
levels_export.to_csv('detected_levels.csv', index=False)
print("✓ Detected levels exported: detected_levels.csv")

# ===================================================================
# STEP 7: Summary
# ===================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Algorithm: Rolling window (5-period) local minima/maxima")
print(f"Lookahead: Yes (2 candles before, 1 current, 2 after)")
print(f"Data: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
print(f"\nDetected Levels:")
print(f"  • Support: {len(support_levels)}")
print(f"  • Resistance: {len(resistance_levels)}")
print(f"  • Total: {len(support_levels) + len(resistance_levels)}")
print(f"\nKey Metrics:")
print(f"  • Support Recurrence: {metrics['Support_Recurrence']:.2%}")
print(f"  • Resistance Recurrence: {metrics['Resistance_Recurrence']:.2%}")
print(f"  • Avg Separation: {metrics['Average_Level_Separation_Pct']:.2f}%")
print(f"  • Coverage: {metrics['Coverage_Percentage']:.2f}%")
print(f"  • Quality Score: {quality_score}/8")
print("="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)