# %%
import numpy as np
from einops import rearrange, reduce, repeat


X = np.arange(1, 13).reshape(4, 3)
print(X)
rearrange(X, 'a b -> b a')
print(X)
# %%
