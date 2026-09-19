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

## What we have done so far (updated on 18 September 11 pm)
- Writing (vibe coding) the scripts to perform clustering, pairing, and tight-binding calculations.
- Identifying lattice structures with <= 100 atoms in a unit cell (m = 2,3,4,5,6,7).
- Testing exponential tight binding parameterization for m = 2,3,4,5,6,7. The hopping parameter is assumed to be isotropic and thus depends only on the distance $R$ between two atoms. 
- We can define the decay length $L$ to parameterize electron localization: low $L$ means highly localized conditions, while large $L$ denotes highly delocalized electrons.
- The hopping parameterization therefore is given by:
$$t = t_0 \exp{(-(R-1)/L)}$$
- Here, $t_0$ is the nearest neighbor hopping parameter for a monolayer square lattice. This parameterization can be considered as the "centered" version of the ordinary exponential.
- The onsite energy terms always appear in the diagonal element of $H(k)$ without any phase factor, and hence it can be seen that changing onsite energy only shifts the band structure & dos vertically.
- We can set the onsite energy to be zero since in this case, we can always normalize the energy value with respect to $t_0$ which implies that we can always set $t_0=1$.
- It seems that as we increase the layer distance, the layers become more decoupled and eventually, there is a fully decoupled limit. This can be seen from the unchanged results for large layer distance (but this has not properly verified).
- For each value of $m$, we currently focus on the variation of decay length and decay distance.

## Notes on the single band fitting (by Anin, updated on 19 September 6 pm)
- If we set $L$ to be large enough such that hoppings between different Moire supercell become probable, I noticed that the band structure retain some particular structure.
- Firstly, there are many densely packed bands around $E=0$. I think this corresponds to the internal band structure on each Moire supercell. As $m$ increases, the supercell becomes larger, and consquently, we can see that more bands become densely packed. 
- Secondly, the lower energy bands (which are less densely packed) exhibit a particular shape. Thus, I have an idea to fit the lowest band into an effective single orbital square lattice tight binding model. Here, I assume that each Moire supercell is represented by a single orbital that has an onsite and hopping parameters up to the next nearest neighbor supercell, i.e.,

$$
E(\vec{k}) = \epsilon_0 - 2t_1(\cos{(\vec{k}\cdot\vec{u}_1)} + \cos{(\vec{k}\cdot\vec{u}_2)}) - 4t_2 \cos{(\vec{k}\cdot\vec{u}_1)} \cos{(\vec{k}\cdot\vec{u}_2)},
$$

- $\vec{u}_1$ and $\vec{u}_2$ denote the primitive lattice vectors of the Moire supercell.
- I asked AI to write a code to implement the fitting, then plot the results for several parameter variations.
- In order to support my assumption that this fitted represent an effective supercell tight binding, I plotted each fitting parameter variation with: twisting angle ($m$ value), exponential decay length, and layer distance.
- One of the most notable result is the fitting parameter variations with the twisting angle: if we set a proper decay length such that the system is neither too localized nor too delocalized, $\epsilon_0$ approaches a constant value while both $t_1$ and $t_2$ decrease for smaller twisting angle. 
- Why is this important? In my opinion, as the supercell becomes bigger for smaller twisting angle, the hopping parameters should be smaller since it will requiere a longer hopping distance. The fitting results show this signature, confirming my initial assumption.
- One might argue that the fitting result is not that accurate. Indeed this is a part that still can be improved. I didn't check how AI implement the fitting algorithm. However, I think this at least gives us a decent rough estimation.
- Another interesting point is that, we may argue that we can expect to obtain a flatband for smaller twisting angles. As both $t_1$ and $t_2$ approach zero for small twisting angles, the bandwidth will get closer to zero.

### Additional comment: all calculations are still manageable on a local computer (no HPC needed). 