"""
Tight-binding band structure and DOS for the CSL bilayer.

A. GENERAL PART  -- reusable machinery: reciprocal lattice, high-symmetry
   points, Hamiltonian construction from pairing.txt, band-structure and
   DOS calculation/plotting. Not normally something you need to edit.

B. SPECIFIC PART -- user-defined physics: hopping(g1, g2, d, ang) and
   smearing(E, Ek), plus their tunable constants. This is the part you're
   expected to replace with your own parameterization.

Conventions locked in during setup:
  - k is always 2D, k = (kx, ky). k_z is never used -- the interlayer
    z-shift (+-d_l) only enters hopping() to compute tB's magnitude/angle,
    never the Bloch phase exp(-i k.r), which always uses the plain in-plane
    (x,y) vector stored in pairing.txt.
  - theta (inside hopping) is the polar angle from the z-axis: 90 degrees
    for a purely in-plane bond, 0/180 degrees for a purely vertical bond.
  - DOS is normalized by the number of k-points (states per unit cell).
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# B. SPECIFIC PART (user-defined formulation)
# =====================================================================

# ---- tunable constants (edit freely) -----------------------------------
E_ON = 0.0       # on-site energy
D_L = 0.1        # interlayer distance (z-separation between r1 and r2)
T0 = 1.0         # hopping prefactor, default exponential model
L_DECAY = 0.4    # decay length, default exponential model
SG = 0.04         # Gaussian smearing width for DOS

def hopping(g1, g2, d, ang_deg):
    """
    tB = hopping(g1, g2, d, ang_deg)  ->  complex

    g1, g2  : point-group labels of the two atoms ('r1' or 'r2')
    d       : in-plane distance between the two atoms (pairing.txt "d")
    ang_deg : in-plane angle of rB - rA, in degrees (pairing.txt "angle")

    On-site term: g1 == g2 and d == 0 -> tB = e_on.
    Otherwise, reconstruct the true 3D displacement r (adding +-d_l along
    z for an inter-layer pair, staying purely in-plane for an intra-layer
    pair), take its magnitude R and its polar angle theta from the z-axis,
    and evaluate the R,theta-dependent model. Default: isotropic
    exponential decay, tB = t0 * exp(-R / L) (ignores theta).
    """
    if g1 == g2 and abs(d) < 1e-9:
        return complex(E_ON)

    ang = np.radians(ang_deg)
    rBA = np.array([d * np.cos(ang), d * np.sin(ang), 0.0])

    if g1 == g2:
        r = rBA
    elif g1 == "r1" and g2 == "r2":
        r = rBA - np.array([0.0, 0.0, D_L])
    elif g1 == "r2" and g2 == "r1":
        r = rBA + np.array([0.0, 0.0, D_L])
    else:
        raise ValueError(f"Unrecognized group pair: {g1!r}, {g2!r}")

    R = float(np.linalg.norm(r))
    theta = 90.0 if R < 1e-12 else float(np.degrees(np.arccos(np.clip(r[2] / R, -1.0, 1.0))))

    # ---- default parameterization: isotropic exponential decay --------

    tB = T0 * np.exp(-R / L_DECAY)
    return complex(tB)


def smearing(E, Ek):
    """
    g(E - Ek) -> real. Default: Gaussian smearing, width SG.
    """
    gauss = (1.0 / (SG * np.sqrt(2 * np.pi))) * np.exp(-(E - Ek) ** 2 / (2 * SG ** 2))
    return gauss


# =====================================================================
# A. GENERAL PART (band structure + DOS machinery)
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
    """Parse pairing.txt into lists of (i, j, group_A, group_B, r, d, angle)."""
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


def precompute_hoppings(rows):
    """Vectorize the parsed rows and evaluate tB once per row (it does not
    depend on k). Returns arrays ready for repeated Hamiltonian builds."""
    i_idx = np.array([row["i"] - 1 for row in rows])
    j_idx = np.array([row["j"] - 1 for row in rows])
    r_arr = np.array([row["r"] for row in rows])
    tB_arr = np.array(
        [hopping(row["g1"], row["g2"], row["d"], row["ang_deg"]) for row in rows],
        dtype=complex,
    )
    n_orb = int(max(i_idx.max(), j_idx.max())) + 1
    return i_idx, j_idx, r_arr, tB_arr, n_orb


def build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb):
    """H(k): for every pairing-file row, H[i,j] += exp(-i k.r) * tB, then
    Hermitized via H = (H + H^dagger)/2."""
    phases = np.exp(-1j * (r_arr @ np.asarray(k, dtype=float)))
    contributions = phases * tB_arr
    H = np.zeros((n_orb, n_orb), dtype=complex)
    np.add.at(H, (i_idx, j_idx), contributions)
    H = (H + H.conj().T) / 2
    return H


def build_kpath(hs_points, labels, n_per_segment=100):
    """Straight segments through hs_points[labels[0]] -> ... -> labels[-1],
    n_per_segment points per segment (both endpoints included), dropping
    the duplicate point at each junction. Returns k_list, k_dist
    (cumulative path length, for physically-spaced plotting), tick_pos
    (k_dist at each labeled point), tick_labels."""
    k_list = [hs_points[labels[0]]]
    k_dist = [0.0]
    tick_pos = [0.0]
    dist_so_far = 0.0

    for seg in range(len(labels) - 1):
        p0, p1 = hs_points[labels[seg]], hs_points[labels[seg + 1]]
        seg_len = np.linalg.norm(p1 - p0)
        t = np.linspace(0.0, 1.0, n_per_segment)[1:]  # drop duplicate start point
        for tt in t:
            k_list.append(p0 + tt * (p1 - p0))
            k_dist.append(dist_so_far + tt * seg_len)
        dist_so_far += seg_len
        tick_pos.append(dist_so_far)

    return np.array(k_list), np.array(k_dist), tick_pos, labels


def compute_band_structure(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, n_per_segment=100):
    hs = high_symmetry_points(b1, b2)
    labels = ["G", "X", "M", "G"]
    k_list, k_dist, tick_pos, tick_labels = build_kpath(hs, labels, n_per_segment)

    bands = np.zeros((len(k_list), n_orb))
    for idx, k in enumerate(k_list):
        H = build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb)
        bands[idx] = np.linalg.eigvalsh(H)

    return k_dist, bands, tick_pos, tick_labels


def plot_band_structure(k_dist, bands, tick_pos, tick_labels, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in range(bands.shape[1]):
        ax.plot(k_dist, bands[:, n], color="blue", linewidth=1.2)
    for pos in tick_pos:
        ax.axvline(pos, color="black", linewidth=0.6)
    label_map = {"G": r"$\Gamma$", "X": "X", "M": "M"}
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([label_map[l] for l in tick_labels])
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylabel("Energy")
    ax.set_title("Band structure")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_dos(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, Emin, Emax, n_k=100, n_E=100):
    """Uniform n_k x n_k sampling of the reciprocal primitive cell
    (fractional coords in [0,1)), Gaussian-smeared finite sum for D(E)
    over n_E points in [Emin, Emax], normalized by the number of
    k-points (states per unit cell)."""
    s = np.arange(n_k) / n_k
    all_evals = []
    for s1 in s:
        for s2 in s:
            k = s1 * b1 + s2 * b2
            H = build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb)
            all_evals.append(np.linalg.eigvalsh(H))
    all_evals = np.array(all_evals)  # shape (n_k*n_k, n_orb)

    N_k = n_k * n_k
    E_grid = np.linspace(Emin, Emax, n_E)
    D = np.array([np.sum(smearing(E, all_evals)) / N_k for E in E_grid])
    return E_grid, D


def plot_dos(E_grid, D, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(D, E_grid, color="blue", linewidth=1.4)
    ax.set_ylabel("Energy")
    ax.set_xlabel("D(E)")
    ax.set_title("Density of states")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Driver
# =====================================================================

if __name__ == "__main__":
    import time

    m = 2

    output_dir = "m=" + str(m) + "/exponential/dl=" + str(D_L) + "/e0=" + str(E_ON) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    def path_out(name):
        return os.path.join(output_dir, name) if output_dir else name

    input_dir = "m=" + str(m) + "/"
    def path_in(name):
        return os.path.join(input_dir, name) if input_dir else name
    
    u1 = np.array([m, 1.0])
    u2 = np.array([-1.0, m])
    b1, b2 = reciprocal_vectors(u1, u2)

    rows = read_pairing_file(path_in("pairing.txt"))
    i_idx, j_idx, r_arr, tB_arr, n_orb = precompute_hoppings(rows)
    print(f"Read {len(rows)} pairs from pairing.txt, {n_orb} orbitals per unit cell")

    t0 = time.time()
    k_dist, bands, tick_pos, tick_labels = compute_band_structure(
        i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, n_per_segment=100
    )
    bs_name = "band_structure_t0=" + str(T0) + "_L=" + str(L_DECAY) + ".png"
    plot_band_structure(k_dist, bands, tick_pos, tick_labels, path_out(bs_name))
    print(f"Wrote band_structure.png ({len(k_dist)} k-points, {time.time()-t0:.2f}s)")

    if bands.min() < 0:
        Emin, Emax = float(1.2 * bands.min()), 1.2 * float(bands.max())
    elif bands.min() > 0:
        Emin, Emax = float(0.8 * bands.min()), 1.2 * float(bands.max())
    print(f"Emin = {Emin:.4f}, Emax = {Emax:.4f}")

    t0 = time.time()
    n_k = 250
    E_grid, D = compute_dos(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, Emin, Emax, n_k=n_k, n_E=100)
    dos_name = "dos_t0=" + str(T0) + "_L=" + str(L_DECAY) + ".png"
    plot_dos(E_grid, D, path_out(dos_name))
    print(f"Wrote dos.png ({n_k}x{n_k} k-mesh, {time.time()-t0:.2f}s)")