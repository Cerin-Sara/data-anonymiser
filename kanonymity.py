import pandas as pd
# import numpy as np

# data = {
#     'Id': [1, 2, 3, 4, 5],
#     'Likes': [10, 20, 30, 40, 50],
#     'Retweets': [10, 20, 30, 40, 50]
# }

# df = pd.DataFrame(data)

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
        for column in quasi_identifiers:
            if column in df.columns:
                block[column] = block[column].sample(frac=1).reset_index(drop=True)  # Shuffle quasi identifier within the block
        shuffled_blocks.append(block)
    
    return pd.concat(shuffled_blocks, ignore_index=True)

# k = 3
# sensitive_attribute = 'Id'
# quasi_identifiers = ['Likes', 'Retweets']
# anonymized_df = k_anonymity(df, k, sensitive_attribute, quasi_identifiers)
# print(anonymized_df)
