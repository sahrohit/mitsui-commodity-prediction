import pandas as pd
import numpy as np
import matplotlib
# Set a non-interactive backend to prevent plt.show() from blocking
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path

# --- 0. Matplotlib Helper Function for Plotting ---
def create_individual_target_plot_matplotlib(df_target, target_id_str, target_name, lag):
    """
    Creates and saves a time series comparison chart for a specific target using Matplotlib.
    """
    print(f"Plotting {target_id_str} ({target_name}) for lag {lag} with Matplotlib...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_target['date_id'], df_target['Actual'], label='Actual', marker='o', linestyle='-')
    plt.plot(df_target['date_id'], df_target['Baseline Model'], label='Baseline Model', marker='x', linestyle='--')
    
    # FIX APPLIED HERE: Using the correct, consistent key
    plt.plot(df_target['date_id'], df_target['Hybrid (LGBM+ARIMA)'], label='Hybrid (LGBM+ARIMA)', marker='s', linestyle=':')
    
    plt.title(f'Actual vs. Predicted (Lag {lag}): {target_name}\n({target_id_str})')
    plt.xlabel('Date ID (Time)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the chart
    chart_filename = f'images/{target_id_str}_timeseries_lag_{lag}.png'
    plt.savefig(chart_filename)
    plt.close() # Close the figure to free up memory
    print(f"Chart saved to: {chart_filename}")

# --- 1. Load Global Files ---
print("--- 1. Loading Global Files ---")
try:
    df_target_info_full = pd.read_csv("dataset/target_pairs.csv")
    df_baseline_full = pd.read_csv("outputs/submission_baseline.csv")
    df_hybrid_full = pd.read_csv("outputs/submission_hybrid_lgbm_arima.csv")
    print("Global files (predictions and target info) loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading global file: {e}")
    print("Cannot proceed without prediction or target info files. Exiting.")
    raise e

# --- 2. Process Each Lag File ---
lags_to_process = [1, 2, 3, 4]
all_mae_summaries = []

print("\n--- 2. Processing Lags ---")

for lag in lags_to_process:
    print(f"\n--- Processing Lag {lag} ---")
    actuals_filename = f"dataset/lagged_test_labels/test_labels_lag_{lag}.csv"
    
    # Check if file exists before trying to load
    if not os.path.exists(actuals_filename):
        print(f"File not found: {actuals_filename}. Skipping this lag.")
        continue
        
    # --- 2a. Load Lag-Specific Data ---
    print(f"Loading {actuals_filename}...")
    df_actuals = pd.read_csv(actuals_filename)
    
    # --- 2b. Filter Targets for this Lag ---
    df_target_info_lag = df_target_info_full[df_target_info_full['lag'] == lag].copy()
    target_cols_lag = df_target_info_lag['target'].tolist()
    actuals_cols_present = [col for col in target_cols_lag if col in df_actuals.columns]
    
    if not actuals_cols_present:
        print(f"No matching target columns found in {actuals_filename} for lag {lag}. Skipping.")
        continue
    
    print(f"Found {len(actuals_cols_present)} target columns for lag {lag}.")

    date_ids = df_actuals['date_id']
    df_actuals_targets = df_actuals[actuals_cols_present]
    
    try:
        df_baseline_targets = df_baseline_full[actuals_cols_present]
        df_hybrid_targets = df_hybrid_full[actuals_cols_present]
    except KeyError as e:
        print(f"Error: Submission files are missing columns for lag {lag}. {e}")
        print("Skipping this lag.")
        continue
        
    # --- 2c. MAE Calculation (Lag {lag}) ---
    print("Calculating MAE...")
    mae_baseline = (df_actuals_targets - df_baseline_targets).abs().mean()
    mae_hybrid = (df_actuals_targets - df_hybrid_targets).abs().mean()

    df_mae = pd.DataFrame({
        'target': actuals_cols_present,
        'baseline_mae': mae_baseline.values,
        'hybrid_mae': mae_hybrid.values
    })
    df_mae['improvement'] = df_mae['baseline_mae'] - df_mae['hybrid_mae']
    df_mae['hybrid_is_better'] = df_mae['improvement'] > 0
    df_mae['lag'] = lag # This is the 'lag' column we want to keep

    # --- FIX APPLIED HERE ---
    # Merge, but drop the redundant 'lag' column from df_target_info_lag first
    # to avoid pandas creating 'lag_x' and 'lag_y'
    df_mae_summary_lag = pd.merge(df_mae, df_target_info_lag.drop(columns=['lag']), on='target', how='left')
    
    all_mae_summaries.append(df_mae_summary_lag)

    # --- 2d. MAE Scatter Plot (Lag {lag}) ---
    print(f"Generating MAE scatter plot for lag {lag} with Matplotlib...")
    
    better = df_mae_summary_lag[df_mae_summary_lag['hybrid_is_better']]
    worse = df_mae_summary_lag[~df_mae_summary_lag['hybrid_is_better']]
    
    max_mae = max(df_mae_summary_lag['baseline_mae'].max(), df_mae_summary_lag['hybrid_mae'].max()) * 1.05
    
    plt.figure(figsize=(10, 10))
    plt.plot([0, max_mae], [0, max_mae], 'r--', label='y=x (Equal MAE)')
    # Plot 'worse' (Baseline Better)
    plt.scatter(worse['baseline_mae'], worse['hybrid_mae'], label='Baseline Better (Above line)', alpha=0.7, c='tab:blue')
    # Plot 'better' (Hybrid Better)
    plt.scatter(better['baseline_mae'], better['hybrid_mae'], label='Hybrid Better (Below line)', alpha=0.7, c='tab:orange')
    
    plt.title(f'Model MAE Comparison (Lag {lag})')
    plt.xlabel('Baseline Model MAE')
    plt.ylabel('Hybrid (LGBM+ARIMA) Model MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max_mae)
    plt.ylim(0, max_mae)
    plt.gca().set_aspect('equal', adjustable='box')
    
    chart_filename = f'images/model_mae_comparison_lag_{lag}.png'
    plt.savefig(chart_filename)
    plt.close()
    print(f"Saved {chart_filename}")

    # --- 2e. Average Time Series Plot (Lag {lag}) ---
    print(f"Generating average time series plot for lag {lag} with Matplotlib...")
    df_avg = pd.DataFrame({
        'date_id': date_ids,
        'Actual': df_actuals_targets.mean(axis=1),
        'Baseline Model': df_baseline_targets.mean(axis=1),
        'Hybrid (LGBM+ARIMA)': df_hybrid_targets.mean(axis=1)
    })
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_avg['date_id'], df_avg['Actual'], label='Actual', marker='o', linestyle='-')
    plt.plot(df_avg['date_id'], df_avg['Baseline Model'], label='Baseline Model', marker='x', linestyle='--')
    plt.plot(df_avg['date_id'], df_avg['Hybrid (LGBM+ARIMA)'], label='Hybrid (LGBM+ARIMA)', marker='s', linestyle=':')
    
    plt.title(f'Actual vs. Predicted: Average of All Targets (Lag {lag})')
    plt.xlabel('Date ID (Time)')
    plt.ylabel('Average Target Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    avg_chart_filename = f'images/average_targets_timeseries_lag_{lag}.png'
    plt.savefig(avg_chart_filename)
    plt.close()
    print(f"Saved {avg_chart_filename}")
    
    # --- 2f. Individual Time Series Plot (Lag {lag}) ---
    print("Generating individual time series plot (example)...")
    first_target = actuals_cols_present[0]
    
    target_name_series = df_target_info_lag.loc[df_target_info_lag['target'] == first_target, 'pair']
    if target_name_series.empty:
        first_target_name = first_target 
    else:
        first_target_name = target_name_series.iloc[0]
    
    df_target_indiv = pd.DataFrame({
        'date_id': date_ids,
        'Actual': df_actuals[first_target],
        'Baseline Model': df_baseline_targets[first_target],
        # FIX APPLIED HERE: Using the correct, consistent key
        'Hybrid (LGBM+ARIMA)': df_hybrid_targets[first_target] 
    })
    
    create_individual_target_plot_matplotlib(df_target_indiv, first_target, first_target_name, lag)

# --- 3. Final Summary Analysis ---
print("\n\n--- 3. Overall Analysis Summary ---")
if not all_mae_summaries:
    print("No lag files were processed. Cannot generate summary.")
else:
    # This concat should now work, as df_mae_summary_lag has a single 'lag' column
    df_all_mae = pd.concat(all_mae_summaries, ignore_index=True)
    
    print("Overall MAE by Lag (Averaged across targets):")
    # This groupby should now work
    df_mae_by_lag_avg = df_all_mae.groupby('lag')[['baseline_mae', 'hybrid_mae']].mean()
    print(df_mae_by_lag_avg)
    
    print("\nOverall MAE (Averaged across all targets and lags):")
    baseline_overall = df_all_mae['baseline_mae'].mean()
    hybrid_overall = df_all_mae['hybrid_mae'].mean()
    print(f"Baseline Overall MAE: {baseline_overall:.6f}")
    print(f"Hybrid Overall MAE:   {hybrid_overall:.6f}")
    
    if baseline_overall < hybrid_overall:
        print("--> The Baseline model performed better overall.")
    else:
        print("--> The Hybrid model performed better overall.")

    # --- 3b. Grouped Bar Chart for MAE by Lag ---
    print("Generating overall MAE by Lag bar chart with Matplotlib...")
    
    df_mae_by_lag_avg.plot(kind='bar', figsize=(10, 6))
    
    plt.title('Overall MAE Comparison by Lag')
    plt.ylabel('Average MAE')
    plt.xlabel('Lag')
    plt.xticks(rotation=0)
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    lag_chart_filename = 'images/overall_mae_by_lag_summary.png'
    plt.savefig(lag_chart_filename)
    plt.close()
    print(f"Saved {lag_chart_filename}")

print("\n--- End of Program ---")