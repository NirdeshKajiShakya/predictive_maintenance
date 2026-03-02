import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('predictive_maintenance.csv')

# ============================================
# FIGURE 3: Class Distribution (Window Display)
# ============================================
fig1, ax1 = plt.subplots(figsize=(8, 5))
counts = df['Machine failure'].value_counts()
bars = ax1.bar(['No Failure (0)', 'Failure (1)'], counts, color=['steelblue', 'coral'])
ax1.set_ylabel('Number of Samples')
ax1.set_title('Figure 3: Class Distribution Before SMOTE\n(3.39% Failure Rate)', fontsize=12)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({count/len(df)*100:.1f}%)',
            ha='center', va='bottom')
plt.tight_layout()
plt.show()  # DISPLAYS WINDOW 1

# ============================================
# FIGURE 4: Correlation Heatmap (Window Display)
# ============================================
plt.figure(figsize=(10, 8))
sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
corr_matrix = df[sensor_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Figure 4: Sensor Correlation Matrix', fontsize=12)
plt.tight_layout()
plt.show()  # DISPLAYS WINDOW 2

# ============================================
# FIGURE 5: Box Plots (Window Display)
# ============================================
fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(sensor_cols):
    df.boxplot(column=col, by='Machine failure', ax=axes[idx])
    axes[idx].set_title(f'{col}\nby Failure Status')
    axes[idx].set_xlabel('Machine Failure (0=No, 1=Yes)')
plt.suptitle('Figure 5: Sensor Value Distributions by Failure Status', fontsize=14)
plt.tight_layout()
plt.show()  # DISPLAYS WINDOW 3

print("All figures displayed. Close each window to continue.")