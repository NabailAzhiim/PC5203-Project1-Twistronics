"""
Bandwidth scaling analysis for the CSL bilayer, built on the same machinery
as band_comparison.py.

Data source -- for each m used, always opens the pre-existing pairing file
    at "m={m}/pairing.txt" (relative to wherever the script runs). This
    script only reads it, never generates it.

B. SPECIFIC PART -- user-defined physics: exponential_centered(g1, g2, d,
    ang_deg, e_on, dl, t0, ldec). Same as band_comparison.py.

A. GENERAL PART -- reciprocal lattice, high-symmetry points, Hamiltonian
    construction from pairing.txt, band-structure calculation along the
    Gamma-X-M-Gamma path. Identical to band_comparison.py.

BANDWIDTH -- for a given m, d_l, L (and fixed e_on, t0):
    Lm = sqrt(m^2 + 1) (the real-space unit-cell side length, |u1| = |u2|).
    Bandwidth W = max - min, sampled along the Gamma-X-M-Gamma path (not a
    full 2D BZ mesh -- confirmed with the user), of:
      - band index 0 alone, for even m (non-degenerate lowest band), or
      - bands 0 and 1 combined, for odd m. The square-lattice-based cell
        used throughout this project is twice the size of the true
        (non-orthogonal) primitive cell for odd m, so what would be one
        non-degenerate band zone-folds into two, split most visibly along
        Gamma-X and M-Gamma.
    We plot W*Lm^2 (not W) against Lm, to cancel the trivial 1/Lm^2
    shrinkage that comes purely from the Brillouin zone getting smaller as
    the unit cell grows, isolating genuine physical trends.

FIGURE -- two independent subplots (no shared axes, no titles):
    (a) W*Lm^2 vs Lm at fixed d_l, one line per L in L_list, for m in m_list.
    (b) W*Lm^2 vs Lm at fixed L, one line per d_l in dl_list, for m in m_list.
    "(a)"/"(b)" sit above each subplot's own y-axis label (same technique
    as the panel letter in band_comparison.py). use_default_font=True
    switches to matplotlib's own default font instead of the APS serif
    style (same option as band_comparison.py).
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pairing_file_path(m):
    """Every m always reads from this fixed, non-generated location."""
    return f"m={m}/pairing.txt"


# =====================================================================
# B. SPECIFIC PART (user-defined formulation)
# =====================================================================

def exponential_centered(g1, g2, d, ang_deg, e_on, dl, t0, ldec):
    """
    tB = exponential_centered(g1, g2, d, ang_deg, e_on, dl, t0, ldec) -> complex
    Identical to band_comparison.py -- see that file for the full docstring.
    """
    if g1 == g2 and abs(d) < 1e-9:
        return complex(e_on)

    ang = np.radians(ang_deg)
    rBA = np.array([d * np.cos(ang), d * np.sin(ang), 0.0])

    if g1 == g2:
        r = rBA
    elif g1 == "r1" and g2 == "r2":
        r = rBA - np.array([0.0, 0.0, dl])
    elif g1 == "r2" and g2 == "r1":
        r = rBA + np.array([0.0, 0.0, dl])
    else:
        raise ValueError(f"Unrecognized group pair: {g1!r}, {g2!r}")

    R = float(np.linalg.norm(r))
    theta = 90.0 if R < 1e-12 else float(np.degrees(np.arccos(np.clip(r[2] / R, -1.0, 1.0))))

    tB = t0 * np.exp(-(R - 1) / ldec)
    return complex(tB)


# =====================================================================
# A. GENERAL PART (band-structure machinery -- identical to band_comparison.py)
# =====================================================================

def reciprocal_vectors(u1, u2):
    omega = u1[0] * u2[1] - u1[1] * u2[0]
    b1 = (2 * np.pi / omega) * np.array([u2[1], -u2[0]])
    b2 = (2 * np.pi / omega) * np.array([-u1[1], u1[0]])
    return b1, b2


def high_symmetry_points(b1, b2):
    return {"G": np.array([0.0, 0.0]), "X": b1 / 2, "M": (b1 + b2) / 2}


_ROW_RE = re.compile(
    r"^\(([-\d.]+),([-\d.]+)\)\t(\d+)\t(\d+)\t(\w+)\t(\w+)\t"
    r"\(([-\d.]+),([-\d.]+)\)\t([-\d.]+)\t([-\d.]+)\s*$"
)


def read_pairing_file(path):
    parsed = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            match = _ROW_RE.match(line)
            if not match:
                raise ValueError(f"Could not parse pairing.txt line: {line!r}")
            _a, _b, i, j, g1, g2, rx, ry, d, ang = match.groups()
            parsed.append({
                "i": int(i), "j": int(j), "g1": g1, "g2": g2,
                "r": np.array([float(rx), float(ry)]),
                "d": float(d), "ang_deg": float(ang),
            })
    return parsed


def precompute_hoppings(rows, params):
    i_idx = np.array([row["i"] - 1 for row in rows])
    j_idx = np.array([row["j"] - 1 for row in rows])
    r_arr = np.array([row["r"] for row in rows])
    tB_arr = np.array(
        [exponential_centered(row["g1"], row["g2"], row["d"], row["ang_deg"], *params) for row in rows],
        dtype=complex,
    )
    n_orb = int(max(i_idx.max(), j_idx.max())) + 1
    return i_idx, j_idx, r_arr, tB_arr, n_orb


def build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb):
    phases = np.exp(-1j * (r_arr @ np.asarray(k, dtype=float)))
    contributions = -phases * tB_arr
    H = np.zeros((n_orb, n_orb), dtype=complex)
    np.add.at(H, (i_idx, j_idx), contributions)
    H = (H + H.conj().T) / 2
    return H


def build_kpath(hs_points, labels, n_per_segment):
    k_list = [hs_points[labels[0]]]
    dist_so_far = 0.0
    for seg in range(len(labels) - 1):
        p0, p1 = hs_points[labels[seg]], hs_points[labels[seg + 1]]
        t = np.linspace(0.0, 1.0, n_per_segment)[1:]
        for tt in t:
            k_list.append(p0 + tt * (p1 - p0))
        dist_so_far += np.linalg.norm(p1 - p0)
    return np.array(k_list)


_STATIC_CACHE = {}


def _get_static_data(m, nkb):
    """Cache per-m data that doesn't depend on e_on/dl/t0/ldec: parsed
    pairing-file rows and the Gamma-X-M-Gamma k-path. Avoids re-reading a
    (potentially large, for high m) pairing.txt file once per curve."""
    key = (m, nkb)
    if key not in _STATIC_CACHE:
        u1 = np.array([m, 1.0])
        u2 = np.array([-1.0, m])
        b1, b2 = reciprocal_vectors(u1, u2)
        hs = high_symmetry_points(b1, b2)
        k_list = build_kpath(hs, ["G", "X", "M", "G"], nkb)
        rows = read_pairing_file(pairing_file_path(m))
        _STATIC_CACHE[key] = (rows, k_list)
    return _STATIC_CACHE[key]


def compute_lowest_bands(m, e_on, dl, t0, ldec, nkb=100):
    """Diagonalize H(k) along Gamma-X-M-Gamma for this (m, e_on, dl, t0,
    ldec) and return the full bands array, shape (n_k, n_orb), ascending."""
    rows, k_list = _get_static_data(m, nkb)
    params = [e_on, dl, t0, ldec]
    i_idx, j_idx, r_arr, tB_arr, n_orb = precompute_hoppings(rows, params)

    bands = np.zeros((len(k_list), n_orb))
    for idx, k in enumerate(k_list):
        H = build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb)
        bands[idx] = np.linalg.eigvalsh(H)
    return bands


# =====================================================================
# BANDWIDTH
# =====================================================================

def bandwidth(m, e_on, dl, t0, ldec, nkb=100):
    """W = max - min along Gamma-X-M-Gamma of the lowest band (even m) or
    the two lowest bands combined (odd m, zone-folded pair). Returns
    (Lm, W*Lm**2)."""
    bands = compute_lowest_bands(m, e_on, dl, t0, ldec, nkb=nkb)
    relevant = bands[:, 0] if m % 2 == 0 else bands[:, 0:2]
    W = float(relevant.max() - relevant.min())
    Lm = float(np.sqrt(m**2 + 1))
    return Lm, W * Lm**2


def bandwidth_curve(m_list, e_on, dl, t0, ldec, nkb=100):
    """W*Lm^2 vs Lm across m_list, at fixed (dl, ldec)."""
    Lm_vals, WLm2_vals = [], []
    for m in m_list:
        Lm, WLm2 = bandwidth(m, e_on, dl, t0, ldec, nkb=nkb)
        Lm_vals.append(Lm)
        WLm2_vals.append(WLm2)
    return np.array(Lm_vals), np.array(WLm2_vals)


# =====================================================================
# FIGURE GENERATION
# =====================================================================

APS_TWO_COLUMN_WIDTH_IN = 7.0  # ~17.8 cm, standard APS full-page figure width

APS_RC_SIZES = {
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
}

APS_RC_FONT = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
}


def _panel_label(ax, text):
    ax.annotate(
        text, xy=(0, 1), xycoords="axes fraction",
        xytext=(-32, 6), textcoords="offset points",
        ha="left", va="bottom", fontsize=12, fontweight="bold",
    )


def _add_top_margin(ax, factor=1.35):
    """Extend the y-axis upper limit so a legend box has clear headroom
    above the highest data point, instead of overlapping the lines.
    Called before legend() so its automatic placement sees the extra room."""
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymin + (ymax - ymin) * factor)


def make_bandwidth_figure(m_list,
                           dl_fixed, L_list,
                           L_fixed, dl_list,
                           e_on, t0,
                           out_path, nkb=100,
                           use_default_font=False,
                           figsize=(APS_TWO_COLUMN_WIDTH_IN, 3.0)):
    """Two independent subplots, no shared axes, no titles:
      (a) W*Lm^2 vs Lm at fixed d_l=dl_fixed, one line per L in L_list.
      (b) W*Lm^2 vs Lm at fixed L=L_fixed, one line per d_l in dl_list.
    m_list is used (scanned) in both subplots. Saves a PDF at out_path,
    creating its parent directory if needed.
    """
    rc = dict(APS_RC_SIZES)
    if not use_default_font:
        rc.update(APS_RC_FONT)

    with plt.rc_context(rc):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)

        for L in L_list:
            Lm_vals, WLm2_vals = bandwidth_curve(m_list, e_on, dl_fixed, t0, L, nkb=nkb)
            axL.plot(Lm_vals, WLm2_vals, "o-", label=f"$L={L}$")
        axL.set_xlabel("$L_m$")
        axL.set_ylabel("$WL_m^2$ (eV)")
        _add_top_margin(axL)
        axL.legend()
        _panel_label(axL, "(a)")

        for dl in dl_list:
            Lm_vals, WLm2_vals = bandwidth_curve(m_list, e_on, dl, t0, L_fixed, nkb=nkb)
            axR.plot(Lm_vals, WLm2_vals, "o-", label=f"$d_l={dl}$")
        axR.set_xlabel("$L_m$")
        axR.set_ylabel("$WL_m^2$ (eV)")
        _add_top_margin(axR)
        axR.legend()
        _panel_label(axR, "(b)")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)
    return out_path


# =====================================================================
# Driver
# =====================================================================

if __name__ == "__main__":
    m_list = [2, 3, 4, 5, 6, 7]

    dl_fixed = 0.1
    L_list = [0.27, 0.30, 0.33, 0.35]

    L_fixed = 0.30
    dl_list = [0.0, 0.1, 0.2, 0.3]

    e_on = 0.0
    t0 = 1.0

    out_path = "figures/bandwidth.pdf"
    use_default_font = True  # True -> matplotlib's default font instead of serif/CM

    make_bandwidth_figure(m_list, dl_fixed, L_list, L_fixed, dl_list,
                           e_on, t0, out_path,
                           use_default_font=use_default_font)
    print(f"Wrote {out_path}")