# %%
import numpy as np
from einops import rearrange, reduce, repeat


X = np.arange(1, 13).reshape(4, 3)
print(X)
X = rearrange(X, 'a b -> b a')
print(X)
# %%

import torch as t
from torch import Tensor
import einops
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")

# %%

# Einops exercises
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    x = t.arange(3, 9)
    return einops.rearrange(x, '(a b) -> a b', a=3)


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

# %%
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    return einops.rearrange(t.arange(1, 7), '(h w) -> h w', h=2, w=3)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

# %%
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, '(weeks days) -> weeks', 'mean', days=7)


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))

#%%
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg_temps = einops.reduce(temps, '(weeks 7) -> weeks', 'mean')

    def sol1():
        avg_temps2 = einops.repeat(avg_temps, 'weeks -> (weeks 7)')
        return temps - avg_temps2

    def sol2():
        tmp = einops.rearrange(temps, '(week day) -> day week', day=7)
        tmp = tmp - avg_temps
        return einops.rearrange(tmp, 'day week -> (week day)')

    return sol2()


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

# %%

# B3 exercises
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    assert len(temps) % 7 == 0
    avg_temps = einops.reduce(temps, '(weeks 7) -> weeks', 'mean')
    std_temps = einops.reduce(temps, '(weeks 7) -> weeks', t.std)
    return (temps - einops.repeat(avg_temps, "w -> (w 7)")) / einops.repeat(std_temps, "weeks -> (weeks 7)")


expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))

# %%

# C exercises
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    norms = t.norm(matrix, dim=1, keepdim=True)
    return matrix / norms


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)


# %%
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    norms = matrix.norm(dim=1, keepdim=True)
    matrix = matrix[:] / norms
    return matrix @ matrix.t()


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)

#%%

# D
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    rands = t.rand(n)
    cum_probs = t.cumsum(probs, dim=0)

    print(rands.shape)
    print(cum_probs.shape)
    return t.searchsorted(cum_probs, rands)


n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)

# %%