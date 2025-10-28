import pandas as pd
import json
import matplotlib.pyplot as plt

# Load null distributions
null_df = pd.read_csv("grangerCopulaNulls.csv")
null_df["null_dist"] = null_df["null_dist"].apply(json.loads)

# Function for a single histogram subplot
def plot_null_subplot(ax, row, bins=30):
    null_dist = row["null_dist"]
    gc_obs = row["GC_test"]

    ax.hist(null_dist, bins=bins, alpha=0.6, density=True,
            color="steelblue", edgecolor="black")
    ax.axvline(gc_obs, color="red", linestyle="--", linewidth=2)
    ax.set_title(f"{row['target_coin']}-{row['target_var']}", fontsize=10)
    ax.set_xlabel("GC under null")
    ax.set_ylabel("Density")

# Function to generate the grouped plots
def plot_grouped_distributions(null_df, predictor_coin, predictor_var, targets=["BTC","ETH","BNB","XRP"]):
    subset = null_df[(null_df["source_coin"] == predictor_coin) & 
                     (null_df["source_var"] == predictor_var)]
    
    target_vars = ["RV", "Log Returns"]  # always target these
    fig, axes = plt.subplots(len(targets), len(target_vars), figsize=(12, 4*len(targets)))
    axes = axes.flatten()
    
    for i, coin in enumerate(targets):
        for j, var in enumerate(target_vars):
            filtered = subset[(subset["target_coin"] == coin) & (subset["target_var"] == var)]
            if filtered.empty:
                print(f"Warning: No data for {coin}-{var} with predictor {predictor_coin}-{predictor_var}")
                axes[i*len(target_vars)+j].set_visible(False)
                continue
            row = filtered.iloc[0]
            plot_null_subplot(axes[i*len(target_vars)+j], row)
    
    fig.suptitle(f"Null Distributions — Predictor: {predictor_coin}-{predictor_var}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Example usage
plot_grouped_distributions(null_df, "USDT", "RV")
plot_grouped_distributions(null_df, "USDT", "LogVolChange")
plot_grouped_distributions(null_df, "USDC", "RV")
plot_grouped_distributions(null_df, "USDC", "LogVolChange")
plot_grouped_distributions(null_df, "DAI", "RV")
plot_grouped_distributions(null_df, "DAI", "LogVolChange")