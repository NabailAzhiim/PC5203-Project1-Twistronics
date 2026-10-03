"""
Neighbor pairing.
Jump to line 58 to set the value of m

Reference block A = R1 (central block). For each selected block B in
R1..R9 (in that order), pair every point in A with every point in B
(N x N = N**2 pairs per selected block, 9 * N**2 pairs in total).

For each pair: vector rB - rA, its magnitude, and its angle relative to
the +x axis (CCW positive, via atan2 -- standard convention already gives
positive angles above the x-axis and negative below).

Outputs:
  pairing.txt         -- pairs in scan order (block R1..R9, then A 1..10, then B 1..10)
  pairing_sorted.txt   -- same pairs, sorted by magnitude d ascending

How to read "pairing.txt" and "pairing_sorted.txt":
-> A is the reference point at the reference block (O)
-> B is the chosen point at the selected block (S)
-> rBA = rB - rA is the displacement vector between A and B
-> For each row,
[Block S position (a,b)] -- [Point A and B index] -- [Point A and B group] -- [rBA vector] -- [rBA magnitude] -- [rBA angle in deg]
-> Positive angles for counter-clockwise rotation, negative angles for clockwise rotation
"""

import numpy as np
from clustering import BLOCKS, cluster_points
import os

def build_pairs(results):
    ref_points = results["R1"]  # list of (group, point) in index order 1..N

    rows = []  # each row: (a, b, idxA, idxB, grpA, grpB, vec, mag, angle_deg)
    for sel_name, (a, b) in BLOCKS.items():
        sel_points = results[sel_name]
        for idxA, (grpA, ptA) in enumerate(ref_points, start=1):
            for idxB, (grpB, ptB) in enumerate(sel_points, start=1):
                vec = ptB - ptA
                mag = float(np.linalg.norm(vec))
                angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
                rows.append((a, b, idxA, idxB, grpA, grpB, vec, mag, angle))
    return rows


def format_row(row):
    a, b, idxA, idxB, grpA, grpB, vec, mag, angle = row
    vec_str = f"({vec[0]:.2f},{vec[1]:.2f})"
    return f"({a},{b})\t{idxA}\t{idxB}\t{grpA}\t{grpB}\t{vec_str}\t{mag:.2f}\t{angle:.2f}\n"


def write_pairing_file(rows, out_path):
    with open(out_path, "w") as f:
        for row in rows:
            f.write(format_row(row))


if __name__ == "__main__":
    m = 0
    results = cluster_points(m)

    outdir = "m=" + str(m) + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pairing_output = os.path.join(outdir, "pairing.txt")
    rows = build_pairs(results)
    write_pairing_file(rows, pairing_output)
    print(f"Wrote pairing.txt with {len(rows)} pairs")

    pairing_sorted_output = os.path.join(outdir, "pairing_sorted.txt")
    rows_sorted = sorted(rows, key=lambda r: r[7])  # sort by raw magnitude (index 7)
    write_pairing_file(rows_sorted, pairing_sorted_output)
    print(f"Wrote pairing_sorted.txt with {len(rows_sorted)} pairs")
