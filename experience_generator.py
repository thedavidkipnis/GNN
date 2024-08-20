import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt




# # Create histogram
# plt.figure(figsize=(10, 6))
# plt.hist(samples, bins=20, edgecolor='black')
# plt.title('Histogram of Custom Skewed Distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

# # Add vertical lines for the 2.5th and 97.5th percentiles
# percentile_2_5 = np.percentile(samples, 2.5)
# percentile_97_5 = np.percentile(samples, 97.5)
# plt.axvline(percentile_2_5, color='r', linestyle='dashed', linewidth=2, label='2.5th percentile')
# plt.axvline(percentile_97_5, color='r', linestyle='dashed', linewidth=2, label='97.5th percentile')

# plt.legend()

# # Print some statistics
# print(f"Min: {np.min(samples):.2f}")
# print(f"Max: {np.max(samples):.2f}")
# print(f"2.5th percentile: {percentile_2_5:.2f}")
# print(f"97.5th percentile: {percentile_97_5:.2f}")

# plt.show()
