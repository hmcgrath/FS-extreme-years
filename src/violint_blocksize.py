
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "D:\\Research\\FS-2dot0\\results\\newtop5\\2000-2023\\all-equalweights\\percentile_justification.csv"
df = pd.read_csv(file_path)

# Calculate uncertainty ranges for wet and dry thresholds for each block size
df['wet_range_b3'] = df['bootstrap_q_wet_hi_b3'] - df['bootstrap_q_wet_lo_b3']
df['wet_range_b5'] = df['bootstrap_q_wet_hi_b5'] - df['bootstrap_q_wet_lo_b5']
df['wet_range_b10'] = df['bootstrap_q_wet_hi_b10'] - df['bootstrap_q_wet_lo_b10']

df['dry_range_b3'] = df['bootstrap_q_dry_hi_b3'] - df['bootstrap_q_dry_lo_b3']
df['dry_range_b5'] = df['bootstrap_q_dry_hi_b5'] - df['bootstrap_q_dry_lo_b5']
df['dry_range_b10'] = df['bootstrap_q_dry_hi_b10'] - df['bootstrap_q_dry_lo_b10']

# Prepare data for plotting
plot_data = pd.DataFrame({
    'Uncertainty Range': pd.concat([
        df['wet_range_b3'], df['wet_range_b5'], df['wet_range_b10'],
        df['dry_range_b3'], df['dry_range_b5'], df['dry_range_b10']
    ], ignore_index=True),
    'Block Size': ['Wet - 3']*len(df) + ['Wet - 5']*len(df) + ['Wet - 10']*len(df) +
                  ['Dry - 3']*len(df) + ['Dry - 5']*len(df) + ['Dry - 10']*len(df)
})

# Plot violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Block Size', y='Uncertainty Range', data=plot_data, inner='quartile', palette='Set2')
plt.title('Comparison of Uncertainty Ranges in Return-Level Thresholds by Block Size')
plt.ylabel('Uncertainty Range (Return-Level Threshold)')
plt.xlabel('Condition and Block Size')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('D:\\Research\\FS-2dot0\\results\\WetDryTrendsPaper\\violint_blocksize.png', dpi=300)
plt.show()
