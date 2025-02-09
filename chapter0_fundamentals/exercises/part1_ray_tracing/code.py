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
        nonlocal rays, segments
        nrays = rays.shape[0]
        nsegments = segments.shape[0]

        # Convert points to 2D
        rays = rays[..., :2]
        segments = segments[..., :2]

        # Reshape rays and segements
        rays = einops.repeat(rays, 'nrays npoints ndim -> nrays nsegments npoints ndim', nsegments=nsegments)
        segments = einops.repeat(segments, 'nsegments npoints ndim -> nrays nsegments npoints ndim', nrays=nrays)
        assert rays.shape == (nrays, nsegments, 2, 2)
        assert segments.shape == (nrays, nsegments, 2, 2)

        O, D = rays[:, :, 0, :], rays[:, :, 1, :]
        L1, L2 = segments[:, :, 0, :], segments[:, :, 1, :]

        mat = t.stack((D, L1 - L2), dim=-1)
        vec = L1 - O
        assert mat.shape == (nrays, nsegments, 2, 2)
        assert vec.shape == (nrays, nsegments, 2)

        dets = t.linalg.det(mat)
        is_singular = dets.abs() < 1e-8
        mat[is_singular] = t.eye(2)

        sol = t.linalg.solve(mat, vec)
        assert sol.shape == (nrays, nsegments, 2)
        u = sol[..., 0]
        v = sol[..., 1]

        is_valid = (u >= 0) & (0 <= v) & (v <= 1)
        assert is_valid.shape == (nrays, nsegments)

        return (~is_singular & is_valid).any(dim=-1)

    return sol2()


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    def sol1():
        rays = t.zeros((num_pixels_y * num_pixels_z, 2, 3))
        ys = t.linspace(-y_limit, y_limit, num_pixels_y)
        zs = t.linspace(-z_limit, z_limit, num_pixels_z)

        # Set directions
        rays[:, 1, 0] = 1
        rays[:, 1, 1] = einops.repeat(ys, 'y -> (y z)', z=num_pixels_z)
        rays[:, 1, 2] = einops.repeat(zs, 'z -> (y z)', y=num_pixels_y)

        return rays
    
    def sol2():
        ys = t.linspace(-y_limit, y_limit, num_pixels_y)
        zs = t.linspace(-z_limit, z_limit, num_pixels_z)

        directions = t.stack(
            (
                t.ones(num_pixels_y * num_pixels_z),
                einops.repeat(ys, "y -> (y z)", z=num_pixels_z),
                einops.repeat(zs, "z -> (y z)", y=num_pixels_y),
            ),
            dim=-1,
        )
        assert directions.shape == (num_pixels_y * num_pixels_z, 3)

        origins = t.zeros(num_pixels_y * num_pixels_z, 3)
        rays = t.stack((origins, directions), dim=1)
        return rays

    return sol2()


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)


# %%

#one_triangle = t.tensor([[0, 0, 0], [4, 0.5, 0], [2, 3, 0]])
#A, B, C = one_triangle
#x, y, z = one_triangle.T
#
#fig: go.FigureWidget = setup_widget_fig_triangle(x, y, z)
#display(fig)
#
#
#@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
#def update(u=0.0, v=0.0):
#    P = A + u * (B - A) + v * (C - A)
#    fig.update_traces({"x": [P[0]], "y": [P[1]]}, 2)
    
# %%

Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    mat = t.stack((-D, (B - A), (C - A)), dim=-1)
    assert mat.shape == (3, 3)

    vec = O - A

    try:
        s, u, v = t.linalg.solve(mat, vec)
    except t.linalg.LinAlgError:
        return False
    
    return s >= 0 and 0 <= u and 0 <= v and u + v <= 1

tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    n = rays.shape[0]

    A, B, C = triangle
    assert A.shape == B.shape == C.shape == (3, )

    O, D = rays[:, 0, :], rays[:, 1, :]
    assert O.shape == D.shape == (n, 3)

    mat = t.stack((-D,
                   einops.repeat((B - A), 'v -> n v', n=n),
                   einops.repeat((C - A), 'v -> n v', n=n)),
                   dim=-1)

    assert mat.shape == (n, 3, 3)

    vec = O - A
    assert vec.shape == (n, 3)

    is_singular = t.linalg.det(mat).abs() < 1e-8
    assert is_singular.shape == (n, )

    mat[is_singular] = t.eye(3)
    sol = t.linalg.solve(mat, vec)
    assert sol.shape == (n, 3)

    s, u, v = sol[:, 0], sol[:, 1], sol[:, 2]
    
    return (~is_singular) & (s >= 0) & (0 <= u) & (0 <= v) & (u + v <= 1)



A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 50
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%

def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size()[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
    assert A.shape == B.shape == C.shape == (NR, 3)

    O, D = rays.unbind(1)
    assert O.shape == D.shape == (NR, 3)

    mat = t.stack([- D, B - A, C - A], dim=-1)
    
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    assert sol.shape == (NR, 3)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%

triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays = rays.shape[0]
    ntriangles = triangles.shape[0]

    # Extract triangles
    A, B, C = einops.repeat(
        triangles,
        'ntriangles pts dims -> nrays ntriangles pts dims',
        nrays=nrays).unbind(dim=2)
    assert A.shape == B.shape == C.shape == (nrays, ntriangles, 3)

    # Extract rays
    O, D = einops.repeat(
        rays,
        'nrays rayPoints dims -> nrays ntriangles rayPoints dims',
        ntriangles=ntriangles
    ).unbind(dim=2)
    assert O.shape == D.shape == (nrays, ntriangles, 3)

    # Solve the equations
    mat = t.stack([-D, B - A, C - A], dim=-1)
    assert mat.shape == (nrays, ntriangles, 3, 3)

    # Guard against equations without a solution
    is_singular = t.linalg.det(mat).abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    assert sol.shape == (nrays, ntriangles, 3)

    # Filter out invalid solutions
    s, u, v = sol.unbind(dim=-1)
    is_valid = ~is_singular & (u >= 0) & (v >= 0) & (u + v <= 1)
    s[~is_valid] = t.tensor(float("inf"))

    # Find the min(s) for each ray
    return s.min(dim=-1).values


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()

# %%

def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return t.tensor([
        [t.cos(theta), 0, t.sin(theta)],
        [0, 1, 0],
        [-t.sin(theta), 0, t.cos(theta)]
    ])

tests.test_rotation_matrix(rotation_matrix)

# %%

def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z] element is distance
    to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)

# %%

def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"], triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    rays = rays.cuda()
    triangles = triangles.cuda()
    return raytrace_mesh(rays, triangles).cpu()


dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)

# %%

