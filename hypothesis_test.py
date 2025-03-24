import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from matplotlib.colors import ListedColormap     # <-- New import
# from sklearn.mixture import BayesianGaussianMixture  # DP‐GMM model (removed if not used)

# Define a minimal angle threshold for directional rotation (in radians)
MIN_ANGLE = 0.01

# Updated movement categories without "Forward Low" and "Backward Low"
MOVEMENT_CATEGORIES = {
    'Stationary': 0,
    'Forward Medium': 1,
    'Forward High': 2,
    'Backward Medium': 3,
    'Backward High': 4,
    'Forward with Clockwise Rotation': 5,
    'Forward with Anticlockwise Rotation': 6,
    'Backward with Clockwise Rotation': 7,
    'Backward with Anticlockwise Rotation': 8,
    'Rotation Left': 9,
    'Rotation Right': 10
}

# Updated color mapping removing "Forward Low" and "Backward Low"
movement_color_mapping = {
    "Stationary": "#66c2a5",
    "Forward Medium": "#fc8d62",
    "Forward High": "#8da0cb",
    "Backward Medium": "#e78ac3",
    "Backward High": "#a6d854",
    "Forward with Clockwise Rotation": "#b3cde3",
    "Forward with Anticlockwise Rotation": "#fdb462",
    "Backward with Clockwise Rotation": "#80b1d3",
    "Backward with Anticlockwise Rotation": "#fb8072",
    "Rotation Left": "#33a02c",
    "Rotation Right": "#e31a1c"
}

# Add IEEE/scientific publication-style formatting (same as plot_model_ranking)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

# Processing functions
def compute_movement_projections(df):
    """Compute movement projections and distances."""
    df['angle_rad'] = np.radians(df['angle'])
    df['forward_movement'] = df['delta_x'] * np.cos(df['angle_rad']) + df['delta_y'] * np.sin(df['angle_rad'])
    df['sideways_movement'] = df['delta_x'] * -np.sin(df['angle_rad']) + df['delta_y'] * np.cos(df['angle_rad'])
    df['distance'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)
    # Signed distance: positive for forward, negative for backward
    df['signed_distance'] = df['forward_movement']
    return df

def compute_thresholds(df):
    """Compute movement thresholds using quantiles from the dataset."""
    return {
        'distance': df['signed_distance'].abs().quantile(0.25),
        'rotation': df['delta_angle'].abs().quantile(0.25),
        'distance_low': df['signed_distance'].abs().quantile(0.33),
        'distance_medium': df['signed_distance'].abs().quantile(0.66)
    }

def classify_movement(row, thresholds):
    """
    Classify movement considering both translation and rotation.
    """
    angle = row['delta_angle']
    no_rotation = abs(angle) < MIN_ANGLE
    dist_abs = abs(row['signed_distance'])
    
    if dist_abs < thresholds['distance']:
        return 'Stationary' if no_rotation else ('Rotation Left' if angle > 0 else 'Rotation Right')
        
    base = "Forward" if row['signed_distance'] > 0 else "Backward"
    return base if no_rotation else f"{base} with {'Anticlockwise' if angle > 0 else 'Clockwise'} Rotation"

def refine_movement_category(row, thresholds):
    """
    Further refine the movement category by splitting pure Forward/Backward movements
    into Medium/High based on thresholds['distance_medium'].
    """
    base_cat = row['movement_category']
    dist = abs(row['signed_distance'])
    if base_cat == 'Forward':
        return "Forward Medium" if dist < thresholds['distance_medium'] else "Forward High"
    elif base_cat == 'Backward':
        return "Backward Medium" if dist < thresholds['distance_medium'] else "Backward High"
    # Optionally, refine rotation categories if desired.
    return base_cat

def correct_phase_values(df):
    """Correct phase values where amplitude is zero."""
    for axis in ['x', 'y', 'z']:
        mask = (df[f'next_amplitude_value_{axis}'] == 0) & (df[f'next_phase_value_{axis}'] != 0)
        df.loc[mask, f'next_phase_value_{axis}'] = 0
    return df

def analyze_combinations(df):
    """Analyze unique combinations and groupings."""
    next_components = ['next_phase_value_x', 'next_phase_value_y', 'next_phase_value_z',
                       'next_amplitude_value_x', 'next_amplitude_value_y', 'next_amplitude_value_z']
    
    unique_before = df[next_components].drop_duplicates().shape[0]
    df = correct_phase_values(df)
    
    unique_combinations = df[next_components].drop_duplicates().reset_index(drop=True)
    combination_to_label = {tuple(row): idx for idx, row in unique_combinations.iterrows()}
    df['label'] = df[next_components].apply(tuple, axis=1).map(combination_to_label)
    df['movement_category_num'] = df['movement_category_refined'].map(MOVEMENT_CATEGORIES)
    
    unique_after = len(combination_to_label)
    
    grouped = df.groupby(['pre_angle_rounded', 'movement_category_refined']).apply(
        lambda g: g[next_components].drop_duplicates().shape[0]
    ).reset_index(name='unique_combos_count')
    
    return unique_before, unique_after, grouped, df

def print_reports(angular_report, distance_report, combinations_data):
    """Print all analysis reports."""
    print("\n=== Angular Delta Report ===")
    print(angular_report)
    print("\n=== Distance Report ===")
    print(distance_report)
    print("\n=== Movement Category Labels (Rule-Based) ===")
    for category, label in MOVEMENT_CATEGORIES.items():
        print(f"{category}: {label}")
    
    unique_before, unique_after, grouped = combinations_data
    print(f"\nTotal unique combinations before correction: {unique_before}")
    print(f"Total unique combinations after correction: {unique_after}")
    
    total_groups = len(grouped)
    groups_with_multiple = (grouped['unique_combos_count'] > 1).sum()
    percent_multiple = groups_with_multiple / total_groups * 100
    print("\nGroups with their unique_combos_count:\n", grouped)
    print(f"\nPercent of groups with multiple combos: {percent_multiple:.2f}%")

def fully_preprocess_data(file_path, epsilon=1e-3):
    """Main preprocessing pipeline."""
    df = pd.read_csv(file_path)
    # Reduce precision of 'signal_state' to 2 decimals if exists
    if "signal_state" in df.columns:
        df["signal_state"] = df["signal_state"].round(2)
    df = (df
            .groupby('episode').apply(lambda x: x.iloc[1:])  # skip first row per episode
            .reset_index(drop=True)
            .pipe(lambda x: x[~((x[['delta_x', 'delta_y', 'delta_angle']].abs() < epsilon).all(axis=1))])
            .pipe(lambda x: x[~((np.abs(x[['delta_x', 'delta_y', 'delta_angle']] -
                                      x[['delta_x', 'delta_y', 'delta_angle']].mean()) >
                                3 * x[['delta_x', 'delta_y', 'delta_angle']].std()).any(axis=1))])
         )
    
    df = compute_movement_projections(df)
    thresholds = compute_thresholds(df)
    
    df['movement_category'] = df.apply(lambda row: classify_movement(row, thresholds), axis=1)
    df['movement_category_refined'] = df.apply(lambda row: refine_movement_category(row, thresholds), axis=1)
    df['pre_angle_rounded'] = df['pre_angle'].round().astype(int)
    
    return df

# Helper function to style axes
def style_axes(ax, spine_width=2):
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(spine_width)

def plot_comparison(df):
    """
    Create a scatter plot showing the rule-based categorization with cluster boundaries,
    complying with the IEEE/scientific publication standards.
    """
    # Build new palette using display labels (standard journal palette)
    legend_label_map = {
        "Rotation Left": "CCW Rotation",
        "Rotation Right": "CW Rotation",
        "Forward Medium": "Forward Medium",
        "Forward High": "Forward High",
        "Forward with Anticlockwise Rotation": "Forward + CCW",
        "Forward with Clockwise Rotation": "Forward + CW",
        "Backward Medium": "Backward Medium",
        "Backward High": "Backward High",
        "Backward with Anticlockwise Rotation": "Backward + CCW",
        "Backward with Clockwise Rotation": "Backward + CW",
        "Stationary": "Stationary"
    }
    legend_order = ["CW Rotation", "CCW Rotation",
                    "Forward Medium", "Forward High",
                    "Forward + CW", "Forward + CCW",
                    "Backward Medium", "Backward High",
                    "Backward + CW", "Backward + CCW",
                    "Stationary"]
    new_palette = {legend_label_map[k]: v for k, v in movement_color_mapping.items()}
    
    df_display = df.copy()
    df_display["movement_category_display"] = df_display["movement_category_refined"].map(legend_label_map)
    
    # Update figure size and dpi to follow publication standards (consistent with plot_model_ranking)
    plt.figure(figsize=(7.5, 4), dpi=300)
    ax = sns.scatterplot(
        data=df_display,
        x='signed_distance',
        y='delta_angle',
        hue='movement_category_display',
        palette=new_palette,
        hue_order=legend_order,
        s=20,  # Reduced marker size (was 40)
        alpha=0.8,
        edgecolor='black'
    )
    # Draw convex boundaries for each cluster, skipping 'Stationary'
    for display_label, color in new_palette.items():
        if display_label == "Stationary":
            continue
        pts = df_display[df_display["movement_category_display"] == display_label][['signed_distance', 'delta_angle']].values
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            plt.plot(pts[verts, 0], pts[verts, 1], color=color, lw=1.0, linestyle='--')
    
    ax.set_xlabel("Signed Distance (Forward/Backward)", fontweight='bold')
    ax.set_ylabel("Angle Change", fontweight='bold')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    leg = ax.get_legend()
    if (leg):
        leg.set_title("")
        for text in leg.get_texts():
            text.set_fontsize(10)
    
    # Style axes using the helper function from file
    style_axes(ax, spine_width=1.0)
    plt.tight_layout()
    plt.savefig("rule_based_categorization.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_signal_pred_basket_counts(df):
    """
    Plot the unique movement_category_num count in 20° baskets,
    using the IEEE/scientific publication standards.
    """
    df = df.copy()
    df = df[df['movement_category_refined'] != 'Stationary']
    df['pre_angle_basket'] = (df['pre_angle'] // 20) * 20
    group_df = df.groupby(['signal_state', 'pre_angle_basket'])['movement_category_num'].nunique().reset_index()
    pivot_df = group_df.pivot(index='signal_state', columns='pre_angle_basket', values='movement_category_num')
    
    # Remove plt.style.use and rely on rcParams for styling
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)
    common_cmap = ListedColormap(list(movement_color_mapping.values()))
    hm = sns.heatmap(pivot_df, annot=False, fmt=".0f", cmap=common_cmap, ax=ax,   # Removed annot to eliminate numbers
                     cbar_kws={"label": "Number of Movements"})
    
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Number of Movements", fontsize=10, fontweight='bold')
    ax.set_xlabel("Angle Basket (Degrees)", fontweight='bold')
    ax.set_ylabel("Signal State", fontweight='bold')
    ax.tick_params(axis='x', labelsize=10, width=1.0)
    ax.tick_params(axis='y', labelsize=10, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.set_xticklabels([str(int(float(label.get_text()))) for label in ax.get_xticklabels()], fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("signal_pred_basket_unique_movement_counts.png", dpi=600, bbox_inches='tight')
    plt.show()

# New: Updated function to print representative quantiles (10%, 50%, 90%) instead of 25%, 50%, 75%.
def print_movement_thresholds(df):
    """
    Prints representative values (10%, 50%, 90% quantiles) for delta_x, delta_y, and delta_angle
    grouped by movement_category_refined.
    """
    print("\n=== Movement Category Representative Deltas ===")
    for category, group in df.groupby('movement_category_refined'):
        q10 = group[['delta_x', 'delta_y', 'delta_angle']].quantile(0.10)
        median = group[['delta_x', 'delta_y', 'delta_angle']].quantile(0.50)
        q90 = group[['delta_x', 'delta_y', 'delta_angle']].quantile(0.90)
        print(f"\nCategory: {category}")
        print("          delta_x      delta_y     delta_angle")
        print(f"   10%:   {q10['delta_x']:.3f}      {q10['delta_y']:.3f}      {q10['delta_angle']:.3f}")
        print(f"   50%:   {median['delta_x']:.3f}      {median['delta_y']:.3f}      {median['delta_angle']:.3f}")
        print(f"   90%:   {q90['delta_x']:.3f}      {q90['delta_y']:.3f}      {q90['delta_angle']:.3f}")

# New combined plotting function
def plot_combined(df):
    """
    Create a combined figure with two subplots:
    (a) Scatter plot of movement types with improved legend.
    (b) Heatmap of unique movement counts with bottom and left spines visible.
    """
    # Build new palette and legend mapping (reuse from plot_comparison)
    legend_label_map = {
        "Rotation Left": "CCW Rotation",
        "Rotation Right": "CW Rotation",
        "Forward Medium": "Forward Medium",
        "Forward High": "Forward High",
        "Forward with Anticlockwise Rotation": "Forward + CCW",
        "Forward with Clockwise Rotation": "Forward + CW",
        "Backward Medium": "Backward Medium",
        "Backward High": "Backward High",
        "Backward with Anticlockwise Rotation": "Backward + CCW",
        "Backward with Clockwise Rotation": "Backward + CW",
        "Stationary": "Stationary"
    }
    legend_order = ["Forward + CW", "Forward + CCW",
                    "Backward + CW", "Backward + CCW",
                    "Forward Medium", "Forward High",
                    "Backward Medium", "Backward High",
                    "CW Rotation", "CCW Rotation",
                    "Stationary"]
    new_palette = {legend_label_map[k]: v for k, v in movement_color_mapping.items()}
    
    # Create a copy with display label column
    df_display = df.copy()
    df_display["movement_category_display"] = df_display["movement_category_refined"].map(legend_label_map)
    
    # Create combined figure with two subplots (a) and (b)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    fig.subplots_adjust(wspace=0.3)
    
    # (a) Movement type scatter plot
    sns.scatterplot(
        data=df_display,
        x='signed_distance',
        y='delta_angle',
        hue='movement_category_display',
        palette=new_palette,
        hue_order=legend_order,
        s=20,  # Reduced marker size
        alpha=0.8,
        edgecolor='black',
        ax=ax_a
    )
    # Draw convex boundaries for each cluster (skip 'Stationary')
    for display_label, color in new_palette.items():
        if display_label == "Stationary":
            continue
        pts = df_display[df_display["movement_category_display"] == display_label][['signed_distance', 'delta_angle']].values
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax_a.plot(pts[verts, 0], pts[verts, 1], color=color, lw=1.0, linestyle='--')
    ax_a.set_xlabel("Signed Distance (Forward/Backward)", fontsize=18, fontweight='bold')  # increased axis title
    ax_a.set_ylabel("Angle Change", fontsize=18, fontweight='bold')
    ax_a.tick_params(axis='x', labelsize=16)
    ax_a.tick_params(axis='y', labelsize=16)
    # Improved legend: title, border, and placed outside the axis
    leg = ax_a.legend(title="Movement Type", fontsize=16, title_fontsize=16, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    leg.get_frame().set_edgecolor('black')
    ax_a.text(-0.15, 1.05, '(a)', transform=ax_a.transAxes, fontsize=16, fontweight='bold')
    ax_a.spines["top"].set_visible(False)  # Hide the top box line
    ax_a.spines["right"].set_visible(False)  # Hide the right box line
    
    # (b) Signal basket heatmap plot
    df_heat = df.copy()
    df_heat = df_heat[df_heat['movement_category_refined'] != 'Stationary']
    df_heat['pre_angle_basket'] = (df_heat['pre_angle'] // 20) * 20
    group_df = df_heat.groupby(['signal_state', 'pre_angle_basket'])['movement_category_num'].nunique().reset_index()
    pivot_df = group_df.pivot(index='signal_state', columns='pre_angle_basket', values='movement_category_num')
    
    common_cmap = ListedColormap(list(movement_color_mapping.values()))
    hm = sns.heatmap(pivot_df, annot=False, fmt=".0f", cmap=common_cmap, ax=ax_b,
                     cbar_kws={"label": "Number of Movements"})
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Number of Movements", fontsize=16, fontweight='bold')
    ax_b.set_xlabel("Angle Basket (Degrees)", fontsize=18, fontweight='bold')  # increased axis title
    ax_b.set_ylabel("Signal State", fontsize=18, fontweight='bold')
    ax_b.tick_params(axis='x', labelsize=16, width=1.0)
    ax_b.tick_params(axis='y', labelsize=16, width=1.0)
    for spine in ax_b.spines.values():
        spine.set_linewidth(1.0)
    # Ensure bottom and left spines are visible
    ax_b.spines["left"].set_visible(True)
    ax_b.spines["bottom"].set_visible(True)
    ax_b.set_xticklabels([str(int(float(label.get_text()))) for label in ax_b.get_xticklabels()], fontsize=10)
    plt.setp(ax_b.get_xticklabels(), rotation=30, ha='right')
    ax_b.text(-0.15, 1.05, '(b)', transform=ax_b.transAxes, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("combined_plots.png", dpi=600, bbox_inches='tight')
    plt.show()

# New function to print summary statistics for positional and angular deltas
def print_key_feature_statistics(df):
    features = ['delta_x', 'delta_y', 'delta_angle']
    print("\n=== Summary Statistics for Positional and Angular Deltas ===")
    for feature in features:
        mean_val = df[feature].mean()
        median_val = df[feature].median()
        std_val = df[feature].std()
        skew_val = df[feature].skew()
        kurt_val = df[feature].kurt()
        print(f"\nFeature: {feature}")
        print(f"  Mean: {mean_val:.3f}")
        print(f"  Median: {median_val:.3f}")
        print(f"  Std: {std_val:.3f}")
        print(f"  Skewness: {skew_val:.3f}")
        print(f"  Kurtosis: {kurt_val:.3f}")

def main():
    """Main execution function."""
    df_cleaned = fully_preprocess_data("RL_data_raw.csv")
    
    # Generate and print reports based on the new rule-based columns.
    angular_report = df_cleaned.groupby('movement_category_refined')['delta_angle'].describe()
    distance_report = df_cleaned.groupby('movement_category_refined')['distance'].describe()
    combinations_data = analyze_combinations(df_cleaned)
    unique_before, unique_after, grouped, df_cleaned = combinations_data
    
    print_reports(angular_report, distance_report, (unique_before, unique_after, grouped))
    
    print("\nSample of rule-based new columns:")
    print(df_cleaned[['movement_category_refined', 'movement_category_num', 'label']].head())
    
    # New: Print movement category IQR deltas table.
    print_movement_thresholds(df_cleaned)
    
    # New: Print key feature summary statistics for positional and angular deltas.
    print_key_feature_statistics(df_cleaned)
    
    # Plot combined figure instead of separate plots
    plot_combined(df_cleaned)
    
    # Remove the last row in each episode since it may not have a label.
    df_cleaned = df_cleaned.groupby('episode').apply(lambda g: g.iloc[:-1]).reset_index(drop=True)
    
    # Save processed data to a new CSV.
    output_file = "RL_data_raw_post_processed.csv"
    df_cleaned.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")

if __name__ == "__main__":
    main()
