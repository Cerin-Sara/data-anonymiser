import numpy as np
import pandas as pd

# data = {
#     'Id': [1, 2, 3, 4, 5],
#     'Likes': [10, 20, 30, 40, 50],
#     'Retweets': [10, 20, 30, 40, 50]
# }

# df = pd.DataFrame(data)

# sensitive_attribute = 'Id'
# quasi_identifiers = ['Likes', 'Retweets']

def k_anonymity(df, k, sensitive_attribute, quasi_identifiers):
    anonymized_df = pd.DataFrame()
    
    for _, row in df.iterrows():
        for _ in range(k):
            anonymized_df = anonymized_df.append(row, ignore_index=True)
    
    shuffled_df = anonymized_df.sample(frac=1).reset_index(drop=True)  # Shuffle the entire DataFrame
    
    shuffled_blocks = []
    block_size = len(df) * k
    
    for i in range(0, len(shuffled_df), block_size):
        block = shuffled_df[i:i+block_size].copy()  # Extract a k-block
        
        # Shuffle the sensitive attribute within the block
        block[sensitive_attribute] = np.random.permutation(block[sensitive_attribute].values)
        
        shuffled_blocks.append(block)
    
    return pd.concat(shuffled_blocks, ignore_index=True)

# df2 = k_anonymity(df, 3, sensitive_attribute, quasi_identifiers)
# print(df2)
