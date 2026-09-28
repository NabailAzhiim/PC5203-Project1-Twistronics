"""
Tight-binding band structure and DOS for the CSL bilayer.

A. GENERAL PART  -- reusable machinery: reciprocal lattice, high-symmetry
   points, Hamiltonian construction from pairing.txt, band-structure and
   DOS calculation/plotting. Not normally something you need to edit.

B. SPECIFIC PART -- user-defined physics: exponential(g1, g2, d, ang) and
   smearing(E, Ek), plus their tunable constants. This is the part you're
   expected to replace with your own parameterization.

Conventions locked in during setup:
  - k is always 2D, k = (kx, ky). k_z is never used -- the interlayer
    z-shift (+-d_l) only enters exponential() to compute tB's magnitude/angle,
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
# def exponential(g1, g2, d, ang_deg, e_on, dl, t0, ldec):
#     """
#     Exponential hopping parameterization

#     tB = exponential(g1, g2, d, ang_deg, eon, dl, t0, ldec)  ->  complex

#     g1, g2  : point-group labels of the two atoms ('r1' or 'r2')
#     d       : in-plane distance between the two atoms (pairing.txt "d")
#     ang_deg : in-plane angle of rB - rA, in degrees (pairing.txt "angle")
#     e_on    : on-site energy
#     dl      : vertical distance between layers
#     t0      : zero-distance hopping parameter
#     ldec    : exponential decay factor

#     Small ldec -> highly localized
#     Large ldec -> highly delocalized

#     On-site term: g1 == g2 and d == 0 -> tB = e_on.
#     Otherwise, reconstruct the true 3D displacement r (adding +-d_l along
#     z for an inter-layer pair, staying purely in-plane for an intra-layer
#     pair), take its magnitude R and its polar angle theta from the z-axis,
#     and evaluate the R,theta-dependent model.
#     """
#     if g1 == g2 and abs(d) < 1e-9:
#         return complex(e_on)

#     ang = np.radians(ang_deg)
#     rBA = np.array([d * np.cos(ang), d * np.sin(ang), 0.0])

#     if g1 == g2:
#         r = rBA
#     elif g1 == "r1" and g2 == "r2":
#         r = rBA - np.array([0.0, 0.0, dl])
#     elif g1 == "r2" and g2 == "r1":
#         r = rBA + np.array([0.0, 0.0, dl])
#     else:
#         raise ValueError(f"Unrecognized group pair: {g1!r}, {g2!r}")

#     R = float(np.linalg.norm(r))
#     theta = 90.0 if R < 1e-12 else float(np.degrees(np.arccos(np.clip(r[2] / R, -1.0, 1.0))))

#     # ---- default parameterization: isotropic exponential decay --------

#     tB = t0 * np.exp(-R / ldec)
#     return complex(tB)

def exponential_centered(g1, g2, d, ang_deg, e_on, dl, t0, ldec):
    """
    Exponential hopping parameterization

    tB = exponential_centered(g1, g2, d, ang_deg, eon, dl, t0, ldec)  ->  complex

    g1, g2  : point-group labels of the two atoms ('r1' or 'r2')
    d       : in-plane distance between the two atoms (pairing.txt "d")
    ang_deg : in-plane angle of rB - rA, in degrees (pairing.txt "angle")
    e_on    : on-site energy
    dl      : vertical distance between layers
    t0      : nearest neighbor hopping for a monolayer square lattice
    ldec    : exponential decay factor

    Small ldec -> highly localized
    Large ldec -> highly delocalized

    On-site term: g1 == g2 and d == 0 -> tB = e_on.
    Otherwise, reconstruct the true 3D displacement r (adding +-d_l along
    z for an inter-layer pair, staying purely in-plane for an intra-layer
    pair), take its magnitude R and its polar angle theta from the z-axis,
    and evaluate the R,theta-dependent model.
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

    # ---- default parameterization: isotropic exponential decay --------
    tB = t0 * np.exp(-(R-1) / ldec)
    return complex(tB)

def smearing(E, Ek, sg):
    """
    Gaussian smearing

    g(E - Ek) -> real.
    """

    gauss = (1.0 / (sg * np.sqrt(2 * np.pi))) * np.exp(-(E - Ek) ** 2 / (2 * sg ** 2))
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


def precompute_hoppings(rows, params):
    """Vectorize the parsed rows and evaluate tB once per row (it does not
    depend on k). Returns arrays ready for repeated Hamiltonian builds."""
    i_idx = np.array([row["i"] - 1 for row in rows])
    j_idx = np.array([row["j"] - 1 for row in rows])
    r_arr = np.array([row["r"] for row in rows])
    tB_arr = np.array(
        [exponential_centered(row["g1"], row["g2"], row["d"], row["ang_deg"], params[0], params[1], params[2], params[3]) for row in rows],
        dtype=complex,
    )
    n_orb = int(max(i_idx.max(), j_idx.max())) + 1
    return i_idx, j_idx, r_arr, tB_arr, n_orb


def build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb):
    """H(k): for every pairing-file row, H[i,j] += exp(-i k.r) * tB, then
    Hermitized via H = (H + H^dagger)/2."""
    phases = np.exp(-1j * (r_arr @ np.asarray(k, dtype=float)))
    contributions = - phases * tB_arr
    H = np.zeros((n_orb, n_orb), dtype=complex)
    np.add.at(H, (i_idx, j_idx), contributions)
    H = (H + H.conj().T) / 2
    return H


def build_kpath(hs_points, labels, n_per_segment):
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


def compute_band_structure(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, nkb):
    hs = high_symmetry_points(b1, b2)
    labels = ["G", "X", "M", "G"]
    k_list, k_dist, tick_pos, tick_labels = build_kpath(hs, labels, nkb)

    bands = np.zeros((len(k_list), n_orb))
    for idx, k in enumerate(k_list):
        H = build_hamiltonian(k, i_idx, j_idx, r_arr, tB_arr, n_orb)
        bands[idx] = np.linalg.eigvalsh(H)

    return k_dist, bands, tick_pos, tick_labels


def plot_band_structure(k_dist, bands, tick_pos, tick_labels, out_path, Erange):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in range(bands.shape[1]):
        ax.plot(k_dist, bands[:, n], color="blue", linewidth=1.2)
    for pos in tick_pos:
        ax.axvline(pos, color="black", linewidth=0.6)
    label_map = {"G": r"$\Gamma$", "X": "X", "M": "M"}
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([label_map[l] for l in tick_labels])
    ax.set_xlim(k_dist[0], k_dist[-1])
    if Erange:
        ax.set_ylim(Erange[0], Erange[1])
    ax.set_ylabel("Energy")
    ax.set_title("Band structure")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_dos(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, Emin, Emax, n_k, n_E, SG):
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
    D = np.array([np.sum(smearing(E, all_evals, SG)) / N_k for E in E_grid])
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

variation = "layer_distance"
SG = 0.1
nkb = 100
nkdos = 100
nEdos = 100
# dl_list = [0.1]
dl_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0]
eon_list = [0.0]
# eon_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 8.0, 16.0]
t0_list = [1.0]
ldec_list = [0.5]
# ldec_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0]
Erg = [-0.8, 3.0]
restrict = False

i = 0
while i < 1:
    if __name__ == "__main__":
        import time

        m = i
        for D_L in dl_list:
            for E_ON in eon_list:
                for T0 in t0_list:
                    for L_DECAY in ldec_list:
                        t0 = time.time()
                        params = [E_ON, D_L, T0, L_DECAY]
                    
                        output_dir = "m=" + str(m) + "/exponential_centered/" + variation + "/"
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
                        i_idx, j_idx, r_arr, tB_arr, n_orb = precompute_hoppings(rows, params)
                        print(f"Read {len(rows)} pairs from pairing.txt, {n_orb} orbitals per unit cell")

                        
                        k_dist, bands, tick_pos, tick_labels = compute_band_structure(
                            i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, nkb
                        )

                        if restrict:
                            Emin, Emax = Erg[0], Erg[1]

                            bs_name = "band_structure_dl=" + str(D_L) + "_eon=" + str(E_ON) + "_t0=" + str(T0) + "_L=" + str(L_DECAY) + "_restricted.png"

                            dos_name = "dos_dl=" + str(D_L) + "_eon=" + str(E_ON) + "_t0=" + str(T0) + "_L=" + str(L_DECAY) + "_restricted.png"
                        else:
                            bs_name = "band_structure_dl=" + str(D_L) + "_eon=" + str(E_ON) + "_t0=" + str(T0) + "_L=" + str(L_DECAY) + ".png"
                            
                            dos_name = "dos_dl=" + str(D_L) + "_eon=" + str(E_ON) + "_t0=" + str(T0) + "_L=" + str(L_DECAY) + ".png"
                            
                            if bands.min() < 0:
                                Emin, Emax = float(1.5 * bands.min()), 1.2 * float(bands.max())
                            elif bands.min() > 0:
                                Emin, Emax = float(0.6 * bands.min()), 1.2 * float(bands.max())
                        
                        plot_band_structure(k_dist, bands, tick_pos, tick_labels, path_out(bs_name), [Emin, Emax])
                        print(f"Wrote band_structure.png ({len(k_dist)} k-points, {time.time()-t0:.2f}s)")

                        t0 = time.time()
                        E_grid, D = compute_dos(i_idx, j_idx, r_arr, tB_arr, n_orb, b1, b2, Emin, Emax, nkdos, nEdos, SG)
                        
                        plot_dos(E_grid, D, path_out(dos_name))
                        print(f"Wrote dos.png ({nkdos}x{nkdos} k-mesh, {time.time()-t0:.2f}s)")
    
    i += 1