"""
APS-style (PRB/PRL) 4-panel band-structure comparison figure for the CSL
bilayer, built from tight_binding_modified_anin.py.

DATA PREPARATION -- regenerates m=<m>/pairing.txt (via the clustering +
    neighbor-pairing procedure from earlier in this project) whenever it
    isn't already present under base_dir. Not something you normally need
    to touch.

B. SPECIFIC PART -- user-defined physics: exponential_centered(g1, g2, d,
    ang_deg, e_on, dl, t0, ldec). This is the part you're expected to
    replace with your own parameterization.

A. GENERAL PART -- reusable machinery: reciprocal lattice, high-symmetry
    points, Hamiltonian construction from pairing.txt, band-structure
    calculation. DOS has been removed -- this script is band-structure
    only.

FIGURE GENERATION -- lays out 4 band-structure panels side by side in
    APS two-column format and saves a PDF.

Conventions kept from the single-figure version:
  - k is always 2D; k_z is never used.
  - theta (inside exponential_centered) is the polar angle from the
    z-axis: 90 deg for a purely in-plane bond, 0/180 deg for a purely
    vertical bond.
  - build_hamiltonian keeps your added minus sign on the hopping
    contribution (contributions = -phases * tB_arr) -- unchanged from
    your upload.

How to use: fill in exactly ONE of m_list, dl_list, ldec_list with 4
values (the parameter that varies across the 4 panels); leave the other
two as single-value lists. eon_list and t0_list must always be
single-value lists.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# DATA PREPARATION
# =====================================================================

BLOCKS = {
    "R1": (0, 0), "R2": (1, 0), "R3": (-1, 0),
    "R4": (0, 1), "R5": (0, -1), "R6": (1, 1),
    "R7": (-1, 1), "R8": (-1, -1), "R9": (1, -1),
}


def _build_point_groups(m, pad=1):
    N = 2 * m + pad
    coords = range(-N, N + 1)
    r1 = [np.array([a, b], dtype=float) for a in coords for b in coords]
    theta = 2 * np.arctan(1.0 / m)
    c, s = np.cos(theta), np.sin(theta)
    Rmat = np.array([[c, -s], [s, c]])
    r2 = [Rmat @ p for p in r1]
    return r1, r2


def _in_block(pt, u1, u2, a, b, det, eps=1e-9):
    rel = pt - (a * u1 + b * u2)
    u = (rel[0] * u2[1] - rel[1] * u2[0]) / det
    v = (u1[0] * rel[1] - u1[1] * rel[0]) / det
    return (-eps <= u < 1 - eps) and (-eps <= v < 1 - eps)


def _cluster_points(m, pad=1):
    u1 = np.array([m, 1.0])
    u2 = np.array([-1.0, m])
    det = u1[0] * u2[1] - u1[1] * u2[0]
    r1, r2 = _build_point_groups(m, pad=pad)
    results = {name: [] for name in BLOCKS}
    for group_name, points in (("r1", r1), ("r2", r2)):
        for pt in points:
            for name, (a, b) in BLOCKS.items():
                if _in_block(pt, u1, u2, a, b, det):
                    results[name].append((group_name, pt))
                    break
    return results


def _build_pairs(results):
    ref_points = results["R1"]
    rows = []
    for sel_name, (a, b) in BLOCKS.items():
        sel_points = results[sel_name]
        for idxA, (grpA, ptA) in enumerate(ref_points, start=1):
            for idxB, (grpB, ptB) in enumerate(sel_points, start=1):
                vec = ptB - ptA
                mag = float(np.linalg.norm(vec))
                angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
                rows.append((a, b, idxA, idxB, grpA, grpB, vec, mag, angle))
    return rows


def _write_pairing_file(rows, out_path):
    with open(out_path, "w") as f:
        for a, b, idxA, idxB, grpA, grpB, vec, mag, angle in rows:
            vec_str = f"({vec[0]:.2f},{vec[1]:.2f})"
            f.write(f"({a},{b})\t{idxA}\t{idxB}\t{grpA}\t{grpB}\t{vec_str}\t{mag:.2f}\t{angle:.2f}\n")


def ensure_pairing_file(m, base_dir="."):
    """Return the path to m=<m>/pairing.txt under base_dir, generating it
    (clustering -> neighbor pairing) if it isn't already there."""
    m_dir = os.path.join(base_dir, f"m={m}")
    pairing_path = os.path.join(m_dir, "pairing.txt")
    if not os.path.exists(pairing_path):
        os.makedirs(m_dir, exist_ok=True)
        rows = _build_pairs(_cluster_points(m))
        _write_pairing_file(rows, pairing_path)
    return pairing_path


# =====================================================================
# B. SPECIFIC PART (user-defined formulation)
# =====================================================================

def exponential_centered(g1, g2, d, ang_deg, e_on, dl, t0, ldec):
    """
    Exponential hopping parameterization

    tB = exponential_centered(g1, g2, d, ang_deg, e_on, dl, t0, ldec) -> complex

    g1, g2  : point-group labels of the two atoms ('r1' or 'r2')
    d       : in-plane distance between the two atoms (pairing.txt "d")
    ang_deg : in-plane angle of rB - rA, in degrees (pairing.txt "angle")
    e_on    : on-site energy
    dl      : vertical distance between layers
    t0      : nearest-neighbor hopping for a monolayer square lattice
    ldec    : exponential decay factor

    Small ldec -> highly localized; large ldec -> highly delocalized.

    On-site term: g1 == g2 and d == 0 -> tB = e_on. Otherwise, reconstruct
    the true 3D displacement r (adding +-d_l along z for an inter-layer
    pair, staying purely in-plane for an intra-layer pair), take its
    magnitude R and its polar angle theta from the z-axis, and evaluate
    the R,theta-dependent model.
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

    # ---- default parameterization: isotropic exponential decay, NN-centered
    tB = t0 * np.exp(-(R - 1) / ldec)
    return complex(tB)


# =====================================================================
# A. GENERAL PART (band-structure machinery)
# =====================================================================

def reciprocal_vectors(u1, u2):
    """General 2D reciprocal-lattice vectors satisfying ai.bj = 2*pi*delta_ij."""
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
    """Parse pairing.txt into a list of row dicts."""
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
    """Vectorize the parsed rows and evaluate tB once per row (it does not
    depend on k). params = [e_on, dl, t0, ldec]."""
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
    """H(k): for every pairing-file row, H[i,j] += -exp(-i k.r) * tB
    (sign kept as in your upload), then Hermitized via H = (H + H^dagger)/2."""
    phases = np.exp(-1j * (r_arr @ np.asarray(k, dtype=float)))
    contributions = -phases * tB_arr
    H = np.zeros((n_orb, n_orb), dtype=complex)
    np.add.at(H, (i_idx, j_idx), contributions)
    H = (H + H.conj().T) / 2
    return H


def build_kpath(hs_points, labels, n_per_segment):
    """Straight segments through hs_points[labels[0]] -> ... -> labels[-1],
    n_per_segment points per segment, dropping the duplicate point at each
    junction. Returns k_list, k_dist (cumulative path length), tick_pos,
    tick_labels."""
    k_list = [hs_points[labels[0]]]
    k_dist = [0.0]
    tick_pos = [0.0]
    dist_so_far = 0.0

    for seg in range(len(labels) - 1):
        p0, p1 = hs_points[labels[seg]], hs_points[labels[seg + 1]]
        seg_len = np.linalg.norm(p1 - p0)
        t = np.linspace(0.0, 1.0, n_per_segment)[1:]
        for tt in t:
            k_list.append(p0 + tt * (p1 - p0))
            k_dist.append(dist_so_far + tt * seg_len)
        dist_so_far += seg_len
        tick_pos.append(dist_so_far)

    return np.array(k_list), np.array(k_dist), tick_pos, labels


def compute_band_structure(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, nkb):
    hs = high_symmetry_points(b1, b2)
    labels = ["G", "X", "M", "G"]
    k_list, k_dist, tick_pos, tick_labels = build_kpath(hs, labels, nkb)

    bands = np.zeros((len(k_list), n_orb))
    for idx, k in enumerate(k_list):
        H = build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb)
        bands[idx] = np.linalg.eigvalsh(H)

    return k_dist, bands, tick_pos, tick_labels


# =====================================================================
# FIGURE GENERATION -- 4-panel APS (PRB/PRL) two-column comparison figure
# =====================================================================

APS_TWO_COLUMN_WIDTH_IN = 7.0  # ~17.8 cm, standard APS full-page figure width

# Always applied: sizing/line-width/PDF-embedding settings appropriate for an
# APS two-column figure. These are not "font" choices, so they apply whether
# or not use_default_font is set.
APS_RC_SIZES = {
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,  # embed scalable (not bitmap Type-3) fonts
}

# Only applied when use_default_font=False (the default): serif/Computer
# Modern typeface matching typical APS typesetting. Set use_default_font=True
# to keep matplotlib's own default font instead.
APS_RC_FONT = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
}

# LaTeX-style symbol for each sweepable parameter's subplot title.
_TITLE_LABEL = {"m": r"$m$", "dl": r"$d_l$", "ldec": r"$L$"}


def _resolve_sweep(m_list, dl_list, ldec_list):
    lists = {"m": m_list, "dl": dl_list, "ldec": ldec_list}
    varying = [name for name, lst in lists.items() if len(lst) > 1]
    if len(varying) != 1:
        raise ValueError(
            "Exactly one of m_list, dl_list, ldec_list must have more than "
            f"one value for a 4-panel comparison (found varying: {varying}). "
            "Fill in 4 values for the parameter you want to compare, and "
            "single-value lists for the other two."
        )
    vary_name = varying[0]
    vary_values = lists[vary_name]
    if len(vary_values) != 4:
        raise ValueError(
            f"{vary_name}_list must have exactly 4 values for a 4-panel "
            f"figure, got {len(vary_values)}."
        )
    return vary_name, vary_values


def make_comparison_figure(m_list, dl_list, ldec_list, eon_list, t0_list,
                            panel_label, out_path, base_dir=".", nkb=100,
                            figsize=(APS_TWO_COLUMN_WIDTH_IN, 2.6),
                            use_default_font=False):
    """Build and save a 4-panel band-structure comparison figure, varying
    exactly one of m, d_l, L across the panels (the other two, plus e_on
    and t0, held fixed). Saves a PDF at out_path.

    use_default_font: False (default) uses a serif/Computer-Modern typeface
    matching typical APS typesetting; True keeps matplotlib's own default
    font instead. Sizing and PDF font embedding are unaffected either way.
    """
    if len(eon_list) != 1:
        raise ValueError("eon_list must have exactly one value.")
    if len(t0_list) != 1:
        raise ValueError("t0_list must have exactly one value.")
    e_on, t0 = eon_list[0], t0_list[0]

    vary_name, vary_values = _resolve_sweep(m_list, dl_list, ldec_list)
    m_fixed = None if vary_name == "m" else (m_list[0] if len(m_list) == 1 else None)
    dl_fixed = None if vary_name == "dl" else (dl_list[0] if len(dl_list) == 1 else None)
    ldec_fixed = None if vary_name == "ldec" else (ldec_list[0] if len(ldec_list) == 1 else None)
    for name, val in (("m_list", m_fixed), ("dl_list", dl_fixed), ("ldec_list", ldec_fixed)):
        if val is None and name != {"m": "m_list", "dl": "dl_list", "ldec": "ldec_list"}[vary_name]:
            raise ValueError(f"{name} must have exactly one value when it is not the varying parameter.")

    panels = []
    for val in vary_values:
        m_here = val if vary_name == "m" else m_fixed
        dl_here = val if vary_name == "dl" else dl_fixed
        ldec_here = val if vary_name == "ldec" else ldec_fixed

        u1 = np.array([m_here, 1.0])
        u2 = np.array([-1.0, m_here])
        b1, b2 = reciprocal_vectors(u1, u2)

        pairing_path = ensure_pairing_file(m_here, base_dir)
        rows = read_pairing_file(pairing_path)
        params = [e_on, dl_here, t0, ldec_here]
        i_idx, j_idx, r_arr, tB_arr, n_orb = precompute_hoppings(rows, params)

        k_dist, bands, tick_pos, tick_labels = compute_band_structure(
            i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, nkb
        )
        panels.append((k_dist, bands, tick_pos, tick_labels))

    # shared y-range: union of all 4 panels' band extrema, + small padding
    Emin = min(p[1].min() for p in panels)
    Emax = max(p[1].max() for p in panels)
    pad = 0.05 * (Emax - Emin)
    Emin, Emax = Emin - pad, Emax + pad

    rc = dict(APS_RC_SIZES)
    if not use_default_font:
        rc.update(APS_RC_FONT)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
        label_map = {"G": r"$\Gamma$", "X": "X", "M": "M"}
        title_label = _TITLE_LABEL[vary_name]

        for ax, (k_dist, bands, tick_pos, tick_labels), val in zip(axes, panels, vary_values):
            for n in range(bands.shape[1]):
                ax.plot(k_dist, bands[:, n], color="blue", linewidth=1.0)
            for pos in tick_pos:
                ax.axvline(pos, color="black", linewidth=0.5)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([label_map[l] for l in tick_labels])
            ax.set_xlim(k_dist[0], k_dist[-1])
            ax.set_ylim(Emin, Emax)
            ax.set_title(f"{title_label} = {val}")

        axes[0].set_ylabel("energy (eV)")

        # Panel label sits directly above the y-axis label (instead of its
        # own separate figure-margin column), so no extra horizontal margin
        # needs to be reserved and the subplots get the freed-up width.
        axes[0].annotate(
            panel_label,
            xy=(0, 1), xycoords="axes fraction",
            xytext=(-32, 6), textcoords="offset points",
            ha="left", va="bottom", fontsize=12, fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(out_path)
        plt.close(fig)
    return out_path


# =====================================================================
# Driver
# =====================================================================

if __name__ == "__main__":
    # Fill in exactly ONE of the following with 4 values -- the parameter
    # you want compared across the 4 panels. Leave the other two as
    # single-value lists.
    m_list = [0]
    dl_list = [0.0, 0.1, 0.5, 2.0]
    # ldec_list = [0.3, 0.5, 1.0, 2.0]
    ldec_list = [0.3]
    eon_list = [0.0]
    t0_list = [1.0]

    panel_label = "(c)"
    out_path = "figures/band_comparison0.pdf"
    base_dir = "."
    use_default_font = True  # True -> matplotlib's default font instead of serif/CM

    make_comparison_figure(m_list, dl_list, ldec_list, eon_list, t0_list,
                            panel_label, out_path, base_dir=base_dir,
                            use_default_font=use_default_font)
    print(f"Wrote {out_path}")