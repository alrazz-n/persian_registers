import glob
import numpy as np

files = sorted(glob.glob("/scratch/project_2005092/nima/token_counts/*.npy"))

all_counts = np.concatenate([np.load(f) for f in files])

print(f"Shards        : {len(files)}")
print(f"Documents     : {len(all_counts):,}")
print(f"Total tokens  : {all_counts.sum():,}")
print(f"Mean          : {all_counts.mean():.1f}")
print(f"Median        : {np.median(all_counts):.1f}")
print(f"Min           : {all_counts.min()}")
print(f"Max           : {all_counts.max()}")

for p in [90, 95, 99]:
    print(f"P{p}: {np.percentile(all_counts, p):.0f}")