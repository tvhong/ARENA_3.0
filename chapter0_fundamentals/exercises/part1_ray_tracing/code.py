# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    def sol1():
        rays = t.zeros((num_pixels, 2, 3))

        # Init origins
        rays[:, 0, :] = 0 # unnecessary as it's already 0

        # Set directions
        y_values = t.linspace(-y_limit, y_limit, num_pixels)
        rays[:, 1, 0] = 1
        rays[:, 1, 1] = y_values
        rays[:, 1, 2] = 0
        return rays

    def sol2():
        rays = t.zeros((num_pixels, 2, 3))
        t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
        rays[:, 1, 0] = 1
        return rays

    return sol2()


rays1d = make_rays_1d(9, 10.0)
#fig = render_lines_with_plotly(rays1d)

# %%

# fig: go.FigureWidget = setup_widget_fig_ray()
# display(fig)
# 
# @interact(v=(0.0, 6.0, 0.01), seed=list(range(10)))
# def update(v=0.0, seed=0):
#     t.manual_seed(seed)
#     L_1, L_2 = t.rand(2, 2)
#     P = lambda v: L_1 + v * (L_2 - L_1)
# 
#     x, y = zip(P(0), P(6))
#     with fig.batch_update():
#         fig.update_traces({"x": x, "y": y}, 0)
#         fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
#         fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)

# %%

def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    ray = ray[:, :2]
    segment = segment[:, :2]

    O, D = ray
    L1, L2 = segment

    A = t.stack([D, L1 - L2], dim=-1)
    B = L1 - O

    assert A.shape == (2, 2)
    assert B.shape == (2, )

    if t.linalg.det(A) == 0:
        return False
    else:
        sol = t.linalg.solve(A, B)
        u, v = sol[0].item(), sol[1].item()
        return u >=0 and 0 <= v <= 1

tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%

def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    def sol1():
        # Convert points to 2D
        n, m = rays.shape[0], segments.shape[0]

        rays = rays[..., :2]
        segments = segments[..., :2]
        assert rays.shape == (n, 2, 2)
        assert segments.shape == (m, 2, 2)

        # Convert batch to n*m
        rays = einops.repeat(rays, 'nrays npoints ndim -> (nrays nsegments) npoints ndim', nsegments=m)
        segments = einops.repeat(segments, 'nsegments npoints ndim -> (nrays nsegments) npoints ndim', nrays=n)
        assert rays.shape == segments.shape == (n * m, 2, 2)

        # Solve equation
        O, D = rays[:, 0, :], rays[:, 1, :]
        L1, L2 = segments[:, 0, :], segments[:, 1, :]
        assert O.shape == D.shape == L1.shape == L2.shape == (n * m, 2)

        A = t.stack([D, L1 - L2], dim=-1)
        B = L1 - O

        assert A.shape == (n * m, 2, 2)
        assert B.shape == (n * m, 2)

        # Filter out singular matrices
        dets = t.linalg.det(A)
        assert dets.shape == (n * m, )
        is_singular = dets < 1e-8
        A[~is_singular] = t.eye(2)

        # Solve the equations
        sol = t.linalg.solve(A, B)
        assert sol.shape == (n * m, 2)

        # Filter the results again where u >= 0 and 0 <= v <= 1
        is_valid = (sol[:, 0] >= 0) & (0 <= sol[:, 1]) & (sol[:, 1] <= 1)
        assert is_valid.shape == (n * m, )

        has_solutions = is_singular & is_valid
        assert has_solutions.shape == (n * m, )

        has_solutions = einops.rearrange(has_solutions, '(nrays nsegments) -> nrays nsegments', nrays=n)
        assert has_solutions.shape == (n, m)

        result = has_solutions.any(dim=-1)
        assert result.shape == (n, )

        return result
    
    def sol2():

        pass

    return sol2()


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
