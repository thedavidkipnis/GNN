import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

def custom_skewed_distribution(size=1):
    a = 6
    loc = 0
    scale = 1
    
    raw_samples = skewnorm.rvs(a, loc, scale, size=size)
    
    min_val, max_val = 0, 25
    target_min, target_max = 2, 10
    
    current_min, current_max = np.percentile(raw_samples, [2.5, 97.5])
    
    scaled_samples = (raw_samples - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    
    final_samples = np.clip(scaled_samples, min_val, max_val)
    
    return final_samples

# Generate 10,000 samples
samples = custom_skewed_distribution(size=10000)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=20, edgecolor='black')
plt.title('Histogram of Custom Skewed Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Add vertical lines for the 2.5th and 97.5th percentiles
percentile_2_5 = np.percentile(samples, 2.5)
percentile_97_5 = np.percentile(samples, 97.5)
plt.axvline(percentile_2_5, color='r', linestyle='dashed', linewidth=2, label='2.5th percentile')
plt.axvline(percentile_97_5, color='r', linestyle='dashed', linewidth=2, label='97.5th percentile')

plt.legend()

# Print some statistics
print(f"Min: {np.min(samples):.2f}")
print(f"Max: {np.max(samples):.2f}")
print(f"2.5th percentile: {percentile_2_5:.2f}")
print(f"97.5th percentile: {percentile_97_5:.2f}")

plt.show()
