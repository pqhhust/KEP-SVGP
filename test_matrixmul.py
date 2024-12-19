import torch

batch_size = 8
num_heads = 12
low_rank = 10
seq_len = 128
hdim = 32
dim = 384

x = torch.randn((batch_size, num_heads, low_rank, seq_len, seq_len))
W = torch.randn((hdim, low_rank))
W_final = torch.randn((dim, dim))
Cov = W.view(1, 1, 1, 1, hdim, low_rank) * x.permute(0, 1, 3, 4, 2).view(batch_size, num_heads, seq_len, seq_len, 1, low_rank) @ W.view(1, 1, 1, 1, hdim, low_rank).permute(0, 1, 2, 3, 5, 4)
det = torch.linalg.slogdet(Cov) 
print(Cov.shape) # batch_size, num_heads, seq_len, seq_len, hdim, hdim
Cov = (W_final.view(1, 1, 1, dim, num_heads, hdim).permute(0, 4, 1, 2, 3, 5) @ Cov @ W_final.view(1, 1, 1, dim, num_heads, hdim).permute(0, 4, 1, 2, 5, 3)).sum(dim=1)
print(Cov.shape)

# import torch

# batch_size = 8
# num_heads = 12
# low_rank = 10
# seq_len = 64
# hdim = 32
# dim = 384

# x = torch.randn((batch_size, num_heads, low_rank, seq_len, seq_len))
# W = torch.randn((hdim, low_rank))
# W_final = torch.randn((dim, dim))
# Cov = torch.einsum('mp, bhijp, np -> bhijmn', W, x.permute(0, 1, 3, 4, 2), W)
# print(Cov.shape) # batch_size, num_heads, seq_len, seq_len, hdim, hdim

# # Reshape W_final to include num_heads and hdim
# W_final_reshaped = W_final.view(num_heads, hdim, dim)  # Shape: (12, 32, 384)

# # Compute the final Covariance using torch.einsum
# # Cov[b, i, j, d, e] = sum_{h, m, n} Cov[b, h, i, j, m, n] * W_final_reshaped[h, m, d] * W_final_reshaped[h, n, e]
# Cov = torch.einsum('bhijmn, hmd, hne -> bijde', Cov, W_final_reshaped, W_final_reshaped)

# print(f'Cov shape: {Cov.shape}')  # Should be (batch_size, seq_len, seq_len, dim, dim)