import torch, numpy as np
import torch.nn.functional as F

# Assume the values ​​of q, k, v
q = np.array([1, 1, 1, 2, 2])
k = np.array([1, 1, 1, 2, 2])
k_transposed = k.reshape(-1, 1)
# Create a fill mask
padding_mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool)

# Calculating attention score
attention_scores = np.dot(q, k_transposed)

# Apply padding mask to attention score
attention_scores.masked_fill_(padding_mask.unsqueeze(1), float('-inf'))

# Printing Results
print("Attention Scores:")
print(attention_scores)

# Calculate attention weights using softmax
attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

# Print attention weights
print("\nAttention Weights:")
print(attention_weights)