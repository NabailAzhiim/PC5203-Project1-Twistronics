"""
Lattice Fourier transform of the twisted square bilayer.

For atomic coordinates r_j = (x_j, y_j) we evaluate directly

    A(k) = sum_j exp(-i k . r_j),        I(k) = |A(k)|^2 ,

for k in the plane (k_z = 0), for
    1. layer A (unrotated),
    2. layer B (rotated CCW by alpha = 2 arctan(n/m)),
    3. the full bilayer, using A_bilayer = A_A + A_B (linearity, Eq. S4).

Outputs (in ./m=<m>_n=<n>/fourier/):
    fourier_bilayer.pdf   single-column figure for Fig. 2 of the main text
    fourier_layers.pdf    two-panel figure with a shared colour scale (Fig. S2)
PNG copies are also written for quick viewing.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PARAMETERS
# ============================================================

m, n = 2, 1          # commensurate pair; the paper uses n = 1

N_REAL = 20          # layer A sites: (p, q) with -N_REAL <= p, q <= N_REAL

N_K = 501            # reciprocal-space grid points per direction
K_MAX = 4 * np.pi    # grid covers -K_MAX <= kx, ky <= K_MAX
# Tip: K_MAX = 5*np.pi with N_K = 751 also puts the coincident Bragg peaks
# of the two layers (e.g. 2*pi*(2,1) for m=2) fully inside the window
# instead of on its edge. Update Eq. (43) and Eq. (S19) if you change this.

MARK_PREDICTED_PEAKS = False   # overlay analytic peak positions (for checking)

SAVE_PNG = True                # also save PNG copies

# Figure sizing for a two-column (revtex) paper, in inches
SINGLE_COLUMN_WIDTH = 3.4
DOUBLE_COLUMN_WIDTH = 7.0

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,        # embed TrueType fonts (journal friendly)
})


# ============================================================
# REAL-SPACE LATTICES
# ============================================================

def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def build_lattices(m, n, N):
    """
    Layer A: integer square lattice (a = 1), a finite (2N+1) x (2N+1) patch.
    Layer B: the same patch rotated CCW by alpha = 2 arctan(n/m) about the
             origin, which is a coincidence site.
    """
    coords = np.arange(-N, N + 1)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    rA = np.column_stack((X.ravel(), Y.ravel())).astype(float)

    alpha = 2 * np.arctan(n / m)
    rB = rA @ rotation(alpha).T

    return rA, rB, alpha


# ============================================================
# FOURIER TRANSFORM
# ============================================================

def lattice_amplitude(points, kx, ky):
    """
    Return the complex amplitude A(kx, ky) = sum_j exp[-i(kx x_j + ky y_j)]
    on the grid, as an array of shape (len(ky), len(kx)).

    Exact atomic coordinates are used (no rasterisation + FFT).
    Looping over ky avoids a huge (Nk, Nk, Natoms) array.
    """
    amp = np.zeros((len(ky), len(kx)), dtype=complex)
    phase_x = np.exp(-1j * np.outer(kx, points[:, 0]))      # (Nkx, Natoms)
    for iy, ky_value in enumerate(ky):
        phase_y = np.exp(-1j * ky_value * points[:, 1])      # (Natoms,)
        amp[iy, :] = phase_x @ phase_y
    return amp


def amplitude_at(points, k):
    """A(k) at a single wavevector k (used for symmetry checks)."""
    return np.exp(-1j * points @ k).sum()


# ============================================================
# ANALYTIC PREDICTIONS
# ============================================================

def predictions(m, n, alpha, verbose=True):
    """Moire and layer reciprocal vectors; returns layer-B primitive vectors."""
    u1 = np.array([m, n], dtype=float)
    u2 = np.array([-n, m], dtype=float)
    omega = u1[0] * u2[1] - u1[1] * u2[0]
    b1 = 2 * np.pi / omega * np.array([u2[1], -u2[0]])
    b2 = 2 * np.pi / omega * np.array([-u1[1], u1[0]])

    g1 = np.array([2 * np.pi, 0.0])
    g2 = np.array([0.0, 2 * np.pi])
    R = rotation(alpha)
    g1B, g2B = R @ g1, R @ g2

    if verbose:
        print("\n------------------------------------")
        print("Analytical predictions")
        print("------------------------------------")
        print(f"(m, n) = ({m}, {n}),  alpha = {np.degrees(alpha):.6f} deg")
        print("Square supercell  u1 =", u1, " u2 =", u2, " area =", omega)
        print("Square-cell reciprocal  b1 =", b1, " b2 =", b2,
              " |b| =", np.linalg.norm(b1))
        if (m - n) % 2 == 0:
            p1, p2 = (u1 + u2) / 2, (u2 - u1) / 2
            print("m, n both odd -> primitive Moire cell is the smaller square")
            print("  (u1+u2)/2 =", p1, " (u2-u1)/2 =", p2,
                  " area =", omega / 2)
            print("  true Moire reciprocal vectors: b1+b2 =", b1 + b2,
                  " b2-b1 =", b2 - b1)
        else:
            print("m + n odd -> the square supercell is primitive")
        print("Layer A reciprocal  g1 =", g1, " g2 =", g2)
        print("Layer B reciprocal  R g1 =", g1B, " R g2 =", g2B)

    return g1B, g2B


def predicted_peaks(g1, g2, kmax):
    """All reciprocal-lattice points h g1 + l g2 inside the plotting window."""
    hmax = int(np.ceil(np.sqrt(2) * kmax / np.linalg.norm(g1))) + 1
    pts = []
    for h in range(-hmax, hmax + 1):
        for l in range(-hmax, hmax + 1):
            G = h * g1 + l * g2
            if np.all(np.abs(G) <= kmax + 1e-9):
                pts.append(G)
    return np.array(pts)


# ============================================================
# SYMMETRY CHECKS
# ============================================================

def check_symmetry(I_grid, points, alpha, n_samples=200, seed=0):
    """
    C4: the grid is symmetric about k = 0 and contains k = 0 (odd N_K), so
        I(C4 k) = I(k) can be checked exactly with np.rot90.
    Mirror across the layer bisector (angle alpha/2) does not map the grid
        onto itself, so it is checked at random wavevectors instead.
    """
    c4_err = np.max(np.abs(np.rot90(I_grid) - I_grid)) / I_grid.max()

    rng = np.random.default_rng(seed)
    r = alpha / 2
    e = np.array([np.cos(r), np.sin(r)])
    mirror = 2 * np.outer(e, e) - np.eye(2)        # reflection across bisector
    ks = rng.uniform(-K_MAX, K_MAX, size=(n_samples, 2))
    Ik = np.array([abs(amplitude_at(points, k))**2 for k in ks])
    Isk = np.array([abs(amplitude_at(points, mirror @ k))**2 for k in ks])
    mir_err = np.max(np.abs(Ik - Isk)) / I_grid.max()

    print("\nSymmetry checks (max deviation / max intensity):")
    print(f"  C4 rotation               : {c4_err:.2e}")
    print(f"  mirror across the bisector: {mir_err:.2e}")


# ============================================================
# PLOTTING
# ============================================================

def draw_panel(ax, I, kx, ky, vmin, vmax, title=None, peaks=None):
    im = ax.imshow(
        np.log1p(I),
        origin="lower",
        extent=[kx[0], kx[-1], ky[0], ky[-1]],
        aspect="equal",
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        interpolation="antialiased",
        rasterized=True,
    )
    ticks = np.arange(-4, 5, 2) * np.pi
    ticks = ticks[np.abs(ticks) <= kx[-1] + 1e-9]
    labels = [r"$0$" if t == 0 else rf"${int(round(t/np.pi))}\pi$" for t in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    if title:
        ax.set_title(title)
    if peaks is not None:
        for P, marker in peaks:
            ax.plot(P[:, 0], P[:, 1], marker, ms=4, mfc="none", mew=0.6)
    return im


def save(fig, path_stem):
    fig.savefig(path_stem + ".pdf", dpi=400, bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(path_stem + ".png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_bilayer(I, kx, ky, path_stem, peaks=None):
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.9))
    lg = np.log1p(I)
    im = draw_panel(ax, I, kx, ky, lg.min(), lg.max(), peaks=peaks)
    fig.colorbar(im, ax=ax, label=r"$\ln(1+I)$", fraction=0.046, pad=0.04)
    save(fig, path_stem)


def plot_layers(IA, IB, kx, ky, path_stem, peaksA=None, peaksB=None):
    """Two panels on a common colour scale, labelled (a) and (b)."""
    lgA, lgB = np.log1p(IA), np.log1p(IB)
    vmin, vmax = min(lgA.min(), lgB.min()), max(lgA.max(), lgB.max())
    fig, axes = plt.subplots(1, 2, figsize=(0.8 * DOUBLE_COLUMN_WIDTH, 2.9),
                             constrained_layout=True)
    draw_panel(axes[0], IA, kx, ky, vmin, vmax, title="(a) Layer $A$",
               peaks=peaksA)
    im = draw_panel(axes[1], IB, kx, ky, vmin, vmax, title="(b) Layer $B$",
                    peaks=peaksB)
    fig.colorbar(im, ax=axes, label=r"$\ln(1+I)$", shrink=0.8, pad=0.02)
    save(fig, path_stem)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    outdir = os.path.join(f"m={m}", "fourier")
    os.makedirs(outdir, exist_ok=True)

    rA, rB, alpha = build_lattices(m, n, N_REAL)
    print("Atoms per layer:", len(rA))
    print(f"Relative twist angle: {np.degrees(alpha):.4f} deg")

    g1B, g2B = predictions(m, n, alpha)

    kx = np.linspace(-K_MAX, K_MAX, N_K)
    ky = np.linspace(-K_MAX, K_MAX, N_K)

    print("\nComputing layer A amplitude...")
    AA = lattice_amplitude(rA, kx, ky)
    print("Computing layer B amplitude...")
    AB = lattice_amplitude(rB, kx, ky)

    IA, IB = np.abs(AA)**2, np.abs(AB)**2
    I_bilayer = np.abs(AA + AB)**2          # linearity of the transform

    print("\nMax ln(1+I): layer A %.3f, layer B %.3f, bilayer %.3f"
          % (np.log1p(IA).max(), np.log1p(IB).max(), np.log1p(I_bilayer).max()))
    print("Expected : ln(1+N^2) = %.3f per layer, ln(1+(2N)^2) = %.3f bilayer"
          % (np.log1p(len(rA)**2), np.log1p((2 * len(rA))**2)))

    check_symmetry(I_bilayer, np.vstack((rA, rB)), alpha)

    peaksA = peaksB = None
    if MARK_PREDICTED_PEAKS:
        peaksA = predicted_peaks(np.array([2*np.pi, 0]),
                                 np.array([0, 2*np.pi]), K_MAX)
        peaksB = predicted_peaks(g1B, g2B, K_MAX)

    plot_bilayer(I_bilayer, kx, ky, os.path.join(outdir, "fourier_bilayer"),
                 peaks=None if peaksA is None
                 else [(peaksA, "wo"), (peaksB, "rs")])
    plot_layers(IA, IB, kx, ky, os.path.join(outdir, "fourier_layers"),
                peaksA=None if peaksA is None else [(peaksA, "wo")],
                peaksB=None if peaksB is None else [(peaksB, "rs")])

    print("\nFigures written to", outdir)
