import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-white')
sns.set_style("white")

# Data
models = ['Base Model', 'Proposed Model']
accuracies = [99.14, 99.55]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

bars = ax.bar(models, accuracies, width=0.5, 
              color=['#E3E3E3', '#2E86C1'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

ax.set_ylim(98.5, 100)
ax.set_ylabel('Accuracy (%)', fontsize=12, labelpad=10)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/docs/figures/res_model_comparison.png', 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none')
plt.show()


#=======================================================================
#=========================== visualization 2 ===========================
#=======================================================================

import pandas as pd

base_model = pd.read_csv("/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/docs/base_model/base_model_timing_results.csv")
proposed_model = pd.read_csv("/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/docs/proposed_model/proposed_model_timing_results.csv")

comparison_df = pd.concat(
    [base_model.rename(columns={"Time (s)": "base_model Time (s)"}),
     proposed_model.rename(columns={"Time (s)": "proposed_model Time (s)"})],
    axis=1
)

print("\n[bold blue]Comparison of Models' Timing[/bold blue]")
print(comparison_df)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# Read the data
base_model = pd.read_csv("/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/docs/base_model/base_model_timing_results.csv")
proposed_model = pd.read_csv("/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/docs/proposed_model/proposed_model_timing_results.csv")

comparison_df = pd.DataFrame({
    'Stage': base_model['Stage'],
    'Base Model': base_model['Time (s)'] / 5,
    'Proposed Model': proposed_model['Time (s)'] / 5
})

# Set the font to a more academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Create separate plots for each stage
stages = comparison_df['Stage'].unique()

for stage in stages:
    # Create figure with adjusted size
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)
    
    # Get data for current stage
    stage_data = comparison_df[comparison_df['Stage'] == stage]
    models = ['Base Model', 'Proposed Model']
    times = [stage_data['Base Model'].iloc[0], stage_data['Proposed Model'].iloc[0]]
    
    # Create horizontal bars with professional colors
    bars = ax.barh(models, times, height=0.4, 
                   color=['#E8E8E8', '#4A90E2'],
                   edgecolor='none',
                   alpha=0.8)
    
    # Customize the plot style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Add subtle grid
    ax.grid(axis='x', linestyle='--', alpha=0.2, color='#666666')
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, 
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}s',
                va='center', 
                ha='left',
                fontsize=10,
                fontweight='normal',
                color='#333333')
    
    # Set title and labels with academic styling
    ax.set_title(stage, pad=20, fontsize=12, fontweight='bold', color='#333333')
    ax.set_xlabel('Time (seconds)', fontsize=10, labelpad=10, color='#333333')
    
    # Rotate y-axis labels for space efficiency
    ax.tick_params(axis='y', labelrotation=45)
    
    # Style the ticks
    ax.tick_params(axis='both', colors='#666666', labelsize=9)
    
    # Add "Lower is Better" annotation
    ax.text(ax.get_xlim()[1], ax.get_ylim()[0] - 0.4,
            '← Lower is Better',
            ha='right',
            va='center',
            fontsize=9,
            style='italic',
            color='#666666')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'timing_comparison_{stage.lower().replace(" ", "_")}.png',
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.2)
    plt.show()

# Display the comparison DataFrame
print("\nComparison of Models' Timing (in seconds):")
print(comparison_df.round(2))




















