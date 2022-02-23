import numpy as np

# generate the data
n_state = 3
n_lam = 3
n_control = 2
# np.random.seed(1)
A = 0.1*np.random.randn(n_state, n_state)
C = np.random.randn(n_state, n_lam)
B = np.random.randn(n_state, n_control)

D = np.random.randn(n_lam, n_state)
E = np.random.randn(n_lam, n_control)
G = 1 * np.random.randn(n_lam, n_lam)
H = np.random.randn(n_lam, n_lam)
lcp_offset = np.random.randn(n_lam)

F = G @ G.T

# form the lcs system
min_sig = min(np.linalg.eigvals(F))
print(min_sig)
print(abs(np.linalg.eigvals(A)))



lcs_mats = {
    'n_state': n_state,
    'n_control': n_control,
    'n_lam': n_lam,
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'G': G,
    'H': H,
    'lcp_offset': lcp_offset,
    'F': F,
    'min_sig': min_sig}

np.save('random_lcs', lcs_mats)
