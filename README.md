# PC5203-Project1-Twistronics
Repository for the PC5203 Course Project 1 - Twistronics

## clustering.py -> points sampling block clustering.

[Jump to the line 161 to set the value of m]

Outputs:
-  clustering.txt         -- list of points in each 3x3 block
-  moire_lattice.txt   -- twisted bilayer lattice patter

How to read "clustering.txt" 
- -> For each row,
- [Block name] -- [Block position (a,b)] -- [Point index] -- [Point group (r1 or r2)] -- [Point coordinate]

## pairing.py -> set up neighbor pairings.

[Jump to line 58 to set the value of m]

Outputs:
-  pairing.txt         -- pairs in scan order (block R1..R9, then A 1..10, then B 1..10)
-  pairing_sorted.txt   -- same pairs, sorted by magnitude d ascending

How to read "pairing.txt" and "pairing_sorted.txt":
- -> A is the reference point at the reference block (O)
- -> B is the chosen point at the selected block (S)
- -> rBA = rB - rA is the displacement vector between A and B
- -> For each row,
- [Block S position (a,b)] -- [Point A and B index] -- [Point A and B group] -- [rBA vector] -- [rBA magnitude] -- [rBA angle in deg]
- -> Positive angles for counter-clockwise rotation, negative angles for clockwise rotation


