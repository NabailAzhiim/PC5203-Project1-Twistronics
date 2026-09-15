# PC5203-Project1-Twistronics
Repository for the PC5203 Course Project 1 - Twistronics

## clustering.py -> points sampling block clustering.

[Jump to the line 161 to set the value of m]

Outputs:
-  clustering.txt         -- list of points in each 3x3 block
-  moire_lattice.txt   -- twisted bilayer lattice patter

How to read "clustering.txt" 
- For each row,

[Block name] -- [Block position (a,b)] -- [Point index] -- [Point group (r1 or r2)] -- [Point coordinate]

## pairing.py -> set up neighbor pairings.

[Jump to line 58 to set the value of m]

Outputs:
-  pairing.txt         -- pairs in scan order (block R1..R9, then A 1..10, then B 1..10)
-  pairing_sorted.txt   -- same pairs, sorted by magnitude d ascending

How to read "pairing.txt" and "pairing_sorted.txt":
- A is the reference point at the reference block (O)
- B is the chosen point at the selected block (S)
- rBA = rB - rA is the displacement vector between A and B
- Positive angles for counter-clockwise rotation, negative angles for clockwise rotation
- For each row,

[Block S position (a,b)] -- [Point A and B index] -- [Point A and B group] -- [rBA vector] -- [rBA magnitude] -- [rBA angle in deg]

## tight_binding.py -> calculate band structure and DOS.
- Divided into: general and specific (user defined) parts.
- General part: general tools to calculate band structure and DOS from a pairing output file.
- Specific part: user defined tight binding parameterization and smearing for DOS calculations, the default method is exponential parameterization with Gaussian smearing.
- line 34-39: tunable parameter for default methods, including on-site energy and layer distance.
- Exponential parameterization: t = t0 exp(-R/L), where the atomic distance R includes the vertical displacement (if any).
- Note that k-path used in the band structure plot is: $\Gamma-X-M-\Gamma$.
- The DOS is calculated by uniformly sampling k-points in the first Brillouin zone with a given k-mesh grid.
- Smearing is used to replace dirac Delta function, you might need to tune up smearing parameters for every different calculation.
-  You can modifiy the k-mesh grid size at line 282.

## What we have done so far
- Writing (vibe coding) the scripts to perform clustering, pairing, and tight-binding calculations.
- Identifying lattice structures with <= 100 atoms in a unit cell (m = 2,3,4,5,6,7).
- Testing exponential tight binding parameterization for m = 2 with a layer distance dl = 0.1.
- It seems the onsite energy only shift the band structure (but this hasn't really verified).
- For the zero onsite energy, it can be seen that weak hopping (small decay length L) causes localization, while strong hopping (large decay length L) effectively closes the band gap (metallic state).