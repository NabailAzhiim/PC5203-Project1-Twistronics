"""
2D coincidence-site-lattice (CSL) block clustering.
Jump to the line 161 to set the value of m

u1 = (m, 1), u2 = (-1, m) are perpendicular, equal-length vectors (u1.u2 = 0),
so R1 (corners 0, u1, u2, u1+u2) is a square tilted by theta/2 = arctan(1/m).

r1: square lattice of integer points (unit spacing).
r2: r1 rotated CCW about the origin by theta = 2*arctan(1/m).

9 blocks R1..R9 = R1 + a*u1 + b*u2 for (a,b) in {-1,0,1}^2 (center, 4 edge
neighbors, 4 diagonal neighbors) -- confirmed to match the problem statement.

Membership test: for a point p and block with base corner P0 = a*u1 + b*u2,
write (p - P0) = u*u1 + v*u2 and solve via the 2D cross product / Cramer's
rule. The point belongs to the block iff 0 <= u < 1 and 0 <= v < 1 (half-open
so that each point maps to exactly one block, since the 9 blocks tile without
overlap).

Sampling range: R1's corner sits AT the origin (not centered on it), so the
3x3 block super-region is not symmetric about (0,0) -- it reaches farther
toward +u1+u2 than toward -u1-u2. A symmetric point range [-2m, 2m] therefore
clips the R6 (+u1+u2) corner. Using [-(2m+1), 2m+1] instead was verified
(for m = 1..4) to give the full sigma = m^2+1 points from each of r1 and r2
in every one of the 9 blocks.

Outputs:
  clustering.txt         -- list of points in each 3x3 block
  moire_lattice.txt   -- twisted bilayer lattice patter

How to read "clustering.txt" -> for each row,
[Block name] -- [Block position (a,b)] -- [Point index] -- [Point group (r1 or r2)] -- [Point coordinate]
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

# ---- block definitions: name -> (a, b) such that block = R1 + a*u1 + b*u2
BLOCKS = {
    "R1": (0, 0),
    "R2": (1, 0),
    "R3": (-1, 0),
    "R4": (0, 1),
    "R5": (0, -1),
    "R6": (1, 1),
    "R7": (-1, 1),
    "R8": (-1, -1),
    "R9": (1, -1),
}


def build_point_groups(m, pad=1):
    """r1: integer square lattice in [-(2m+pad), 2m+pad]^2.
    r2: r1 rotated CCW by theta = 2*arctan(1/m) about the origin."""
    N = 2 * m + pad
    coords = range(-N, N + 1)
    r1 = [np.array([a, b], dtype=float) for a in coords for b in coords]

    if (m == 0):
        theta = np.pi
    else:
        theta = 2 * np.arctan(1.0 / m)
    c, s = np.cos(theta), np.sin(theta)
    Rmat = np.array([[c, -s], [s, c]])
    r2 = [Rmat @ p for p in r1]
    return r1, r2


def in_block(pt, u1, u2, a, b, det, eps=1e-9):
    """Fractional-coordinate point-in-parallelogram test (half-open)."""
    rel = pt - (a * u1 + b * u2)
    u = (rel[0] * u2[1] - rel[1] * u2[0]) / det
    v = (u1[0] * rel[1] - u1[1] * rel[0]) / det
    return (-eps <= u < 1 - eps) and (-eps <= v < 1 - eps)


def cluster_points(m, pad=1):
    u1 = np.array([m, 1.0])
    u2 = np.array([-1.0, m])
    det = u1[0] * u2[1] - u1[1] * u2[0]  # = m^2 + 1

    r1, r2 = build_point_groups(m, pad=pad)

    results = {name: [] for name in BLOCKS}
    for group_name, points in (("r1", r1), ("r2", r2)):
        for pt in points:
            for name, (a, b) in BLOCKS.items():
                if in_block(pt, u1, u2, a, b, det):
                    results[name].append((group_name, pt))
                    break  # blocks tile without overlap -> at most one match
    return results


def write_clustering_file(results, out_path):
    with open(out_path, "w") as f:
        for name, (a, b) in BLOCKS.items():
            for idx, (grp, pt) in enumerate(results[name], start=1):
                f.write(f"{name}\t({a},{b})\t{idx}\t{grp}\t({pt[0]:.6f},{pt[1]:.6f})\n")


def plot_pattern(m, out_path, N=None):
    """Plot r1 (blue), r2 (red), the CSL coincidence sites (green rings), and
    the 9 blocks (R1 outlined thick/black, R2-R9 thin/gray). Purely a visual
    rendering of the same lattice + block math used above -- nothing here
    feeds back into clustering.txt or pairing.txt."""
    

    u1 = np.array([m, 1.0])
    u2 = np.array([-1.0, m])
    if (m == 0):
        theta = np.pi
    else:
        theta = 2 * np.arctan(1.0 / m)
    c, s = np.cos(theta), np.sin(theta)
    Rmat = np.array([[c, -s], [s, c]])

    if N is None:
        N = 2 * m + 4  # comfortably covers the 9-block region for the plot

    rng = range(-N, N + 1)
    r1 = np.array([[a, b] for a in rng for b in rng], dtype=float)
    r2 = (Rmat @ r1.T).T

    # Coincidence sites: a*u1 + b*u2 for integers a,b within the plot range
    lattice = []
    for a in range(-N, N + 1):
        for b in range(-N, N + 1):
            p = a * u1 + b * u2
            if -N <= p[0] <= N and -N <= p[1] <= N:
                lattice.append(p)
    lattice = np.array(lattice)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(r1[:, 0], r1[:, 1], s=18, color="#3B6FCC", label="Layer A", zorder=2)
    ax.scatter(r2[:, 0], r2[:, 1], s=18, color="#CC5A3B", label="Layer B", zorder=2, alpha=0.85)

    # Original square CSL coincidence sites and 3x3 block grid.
    if len(lattice) > 0:
        ax.scatter(lattice[:, 0], lattice[:, 1], s=52, facecolors="none",
                   edgecolors="#2E8B57", linewidths=1.4,
                   label=r"CSL sites ($\mathbf{u}_1,\mathbf{u}_2$)", zorder=4)

    for name, (a, b) in BLOCKS.items():
        corners = np.array([a * u1 + b * u2, (a + 1) * u1 + b * u2,
                             (a + 1) * u1 + (b + 1) * u2, a * u1 + (b + 1) * u2,
                             a * u1 + b * u2])
        lw = 2.5 if name == "R1" else 1.2
        col = "#111111" if name == "R1" else "#888888"
        ax.plot(corners[:, 0], corners[:, 1], color=col, linewidth=lw, zorder=1)

    # For odd m, the square u-cell is non-primitive.  Overlay the true
    # primitive cell generated by v1,v2, using the same origin and the same
    # 3x3 block convention.  Orange markers show the corresponding primitive
    # coincidence-site lattice.  This is visualization only and does not
    # modify clustering.txt or the u1/u2-based clustering convention.
    if m % 2 == 1:
        v1 = 0.5 * (u1 + u2)
        v2 = 0.5 * (-u1 + u2)

        primitive_lattice = []
        for a in range(-2 * N, 2 * N + 1):
            for b in range(-2 * N, 2 * N + 1):
                p = a * v1 + b * v2
                if -N <= p[0] <= N and -N <= p[1] <= N:
                    primitive_lattice.append(p)
        primitive_lattice = np.array(primitive_lattice)

        if len(primitive_lattice) > 0:
            ax.scatter(primitive_lattice[:, 0], primitive_lattice[:, 1],
                       s=34, marker="o", facecolors="none",
                       edgecolors="#E67E22", linewidths=1.25,
                       label=r"Primitive CSL sites ($\mathbf{v}_1,\mathbf{v}_2$)",
                       zorder=5)

        for name, (a, b) in BLOCKS.items():
            corners = np.array([a * v1 + b * v2,
                                (a + 1) * v1 + b * v2,
                                (a + 1) * v1 + (b + 1) * v2,
                                a * v1 + (b + 1) * v2,
                                a * v1 + b * v2])
            lw = 2.5 if name == "R1" else 1.2
            col = "#D35400" if name == "R1" else "#F5B041"
            ax.plot(corners[:, 0], corners[:, 1], color=col,
                    linewidth=lw, zorder=3)

    # bounding box of the 9-block region (derived from the block corner
    # extents), plus a small margin for readability
    # Plot limits: include the full 3x3 u-cell grid and, for odd m,
    # the full 3x3 primitive v-cell grid, with extra margin.
    def grid_corner_cloud(a1, a2):
        pts = []
        for a, b in BLOCKS.values():
            pts.extend([
                a*a1 + b*a2,
                (a+1)*a1 + b*a2,
                (a+1)*a1 + (b+1)*a2,
                a*a1 + (b+1)*a2,
            ])
        return np.asarray(pts)

    extent_pts = [grid_corner_cloud(u1, u2)]
    if m % 2 == 1:
        extent_pts.append(grid_corner_cloud(v1, v2))

    extent_pts = np.vstack(extent_pts)
    margin = max(2.0, 0.25*np.linalg.norm(u1))

    x_min = extent_pts[:, 0].min() - margin
    x_max = extent_pts[:, 0].max() + margin
    y_min = extent_pts[:, 1].min() - margin
    y_max = extent_pts[:, 1].max() + margin

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_title(f"m = {m}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    m = 7
    results = cluster_points(m)
    counts = {name: len(v) for name, v in results.items()}
    print("Points per block:", counts)
    print("All equal:", len(set(counts.values())) == 1)

    outdir = "m=" + str(m) + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    clustering_output = os.path.join(outdir, "clustering.txt")
    write_clustering_file(results, clustering_output)
    print("Wrote clustering.txt")

    figure_output = os.path.join(outdir, "moire_lattice.pdf")
    plot_pattern(m, figure_output)
    print("Plotted moire_lattice")