import pandas as pd
import numpy as np

# Load the original large dataset
print("Loading original dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Original dataset shape: {df.shape}")

# Get all fraud cases (Class=1)
fraud_df = df[df['Class'] == 1]
print(f"Number of fraud cases: {len(fraud_df)}")

# Get all legitimate cases (Class=0)
legit_df = df[df['Class'] == 0]
print(f"Number of legitimate cases: {len(legit_df)}")

# Split legitimate transactions into two parts while keeping all fraud cases together
# This ensures both parts have fraud cases for better training
legit_part1 = legit_df.iloc[:len(legit_df)//2]
legit_part2 = legit_df.iloc[len(legit_df)//2:]

# Create part 1 with all fraud cases and first half of legitimate cases
part1 = pd.concat([legit_part1, fraud_df])
part1 = part1.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
print(f"Part 1 shape: {part1.shape}")

# Create part 2 with second half of legitimate cases
part2 = legit_part2.reset_index(drop=True)
print(f"Part 2 shape: {part2.shape}")

# Save both parts
part1.to_csv('creditcard_part1.csv', index=False)
part2.to_csv('creditcard_part2.csv', index=False)

print("Dataset successfully split into creditcard_part1.csv and creditcard_part2.csv")
print("Part 1 contains all fraud cases and half of legitimate cases")
print("Part 2 contains the other half of legitimate cases")