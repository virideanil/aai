> Reviewed project snapshot, 14 October 2026. Written by Claude (Fable) and independently reviewed by Codex. The source manuscript is retained locally. Two phrases were qualified in this export: computer brackets have rational endpoints in the stated corollary, and transcendence means that pi is not algebraic. These are classical mathematical results with the scope stated below. The [operating design](wiring.md#pi-shared-structure-and-continuing-refinement) explains their role in nano’ai; the [original demonstration](../examples/trit-computer.html) is included as source.

# What can be proven about π in the modified ternary

Written 13 October 2026 by Claude (Fable) at the owner's request ("give me the proof for π"), for independent verification by Codex. Everything below is either proven here in full or cited as classical with its author. Nothing is claimed beyond the theorems.

## 0. Definitions and the modified ternary

Let π be the standard constant of analysis: twice the first positive zero of the cosine function (equivalently the period of sine divided by 2); this is the π of Theorems 3 and 4. Let D = {(x, y) : x ≥ 0, y ≥ 0, x² + y² ≤ 1} be the closed quarter of the unit disk. D is Jordan measurable: its boundary consists of two segments and a circular arc, a curve of finite length, hence of plane content zero; so its area exists.

**Lemma 0.1.** area(D) = π/4.

Proof. area(D) = ∫₀¹ √(1 − x²) dx. Substitute x = sin θ, 0 ≤ θ ≤ π/2, dx = cos θ dθ: the integral becomes ∫₀^{π/2} cos² θ dθ = ∫₀^{π/2} (1 + cos 2θ)/2 dθ = π/4. ∎ (This identifies the geometric quarter-disk area with the analytic π, so one π is used throughout; the circumference definition is Archimedes' theorem and is not needed.)

Fix an integer n ≥ 1 and cut the unit square into n² closed cells C(i, j) = [i/n, (i+1)/n] × [j/n, (j+1)/n], 0 ≤ i, j ≤ n − 1. The near corner of C(i, j) is (i/n, j/n), the far corner is ((i+1)/n, (j+1)/n). Judge each cell with integers only:

- yes if (i+1)² + (j+1)² ≤ n²;
- no if i² + j² ≥ n²;
- both otherwise (the owner's third value: the cell holds 0 and 1 together; the page calls it maybe).

Let Y(n), M(n), N(n) be the numbers of yes, both and no cells. These are the only inputs; every judgment is an integer comparison.

## 1. Theorem 1 (the bracket, and why "both" is the right name)

**Lemma 1.1.** A yes cell lies entirely in D. A no cell meets D at most on the unit circle, a set of area zero. A both cell contains a point strictly inside the disk and a point strictly outside it.

Proof. Let (x, y) ∈ C(i, j). Then i/n ≤ x ≤ (i+1)/n and j/n ≤ y ≤ (j+1)/n. If the cell is yes, x² + y² ≤ ((i+1)² + (j+1)²)/n² ≤ 1, so (x, y) ∈ D. If the cell is no, x² + y² ≥ (i² + j²)/n² ≥ 1, so (x, y) is outside the open disk; the cell meets D only where x² + y² = 1, which is a subset of the circle, of area zero. If the cell is both, its near corner satisfies (i/n)² + (j/n)² < 1 (strictly inside) and its far corner satisfies ((i+1)/n)² + ((j+1)/n)² > 1 (strictly outside). ∎

**Theorem 1.** For every n ≥ 1,

    4 · Y(n) / n²  ≤  π  ≤  4 · (Y(n) + M(n)) / n².

Proof. The n² cells have area 1/n² each and overlap only on their edges, a set of area zero. By Lemma 1.1 the union of the yes cells is contained in D, so area(D) ≥ Y(n)/n². Every point of D lies in some cell, and by Lemma 1.1 that cell is not a no cell unless the point is on the circle (area zero); so D is contained in the union of the yes and both cells together with a set of area zero, and area(D) ≤ (Y(n) + M(n))/n². Multiply both inequalities by 4. ∎

Remark (the owner's reading made exact). The third value here is not "unknown": a both cell is proven to contain inside and outside points. The trit is a projection of that structured fact, which is what the owner asked the modified ternary to be.

## 2. Theorem 2 (the both cells are few because they share one boundary)

**Theorem 2.** M(n) ≤ 3n for every n ≥ 1. Consequently the width of the bracket satisfies 4 · M(n)/n² ≤ 12/n, and

    π = lim_{n→∞} 4 · Y(n) / n²,

and π is the only real number lying in the bracket of Theorem 1 for every n.

Proof. For 0 ≤ k ≤ n put s_k = √(n² − k²), a real number with s_0 = n, s_n = 0, strictly decreasing in k. The cell C(i, j) is both exactly when i² + j² < n² and (i+1)² + (j+1)² > n², that is, when j < s_i and j + 1 > s_{i+1}; so the both cells of column i are the integers j in the open interval (s_{i+1} − 1, s_i). An open interval (α, β) contains at most β − α + 1 integers. Hence column i holds at most s_i − s_{i+1} + 2 both cells, and summing over i = 0, …, n − 1 the differences telescope:

    M(n) ≤ (s_0 − s_n) + 2n = n + 2n = 3n.

The width is 4M(n)/n² ≤ 12/n → 0, and since π lies in every bracket (Theorem 1), 4Y(n)/n² → π. If a real x lies in every bracket then |x − π| ≤ 12/n for all n, so x = π. ∎

Remark (the owner's identity point in geometry). The both cells are not independent unknowns: they are the cells met by one shared curve, the arc, and that shared source is why their number is linear in n while the cells number n². Two maybes that share a source are bound together; the count proves it.

## 3. Theorem 3 (no finite rational stage ever closes the band)

**Theorem 3 (Lambert 1761; proof by Niven 1947).** π is irrational.

Proof. Suppose π = a/b with positive integers a, b. For an integer n ≥ 1 define

    f(x) = xⁿ (a − bx)ⁿ / n!,     F(x) = f(x) − f''(x) + f⁗(x) − ⋯ + (−1)ⁿ f^(2n)(x).

(i) n! · f(x) is a polynomial with integer coefficients whose powers of x run from n to 2n. So for every k, f^(k)(0) is an integer: it is 0 for k < n, and for n ≤ k ≤ 2n it equals (k!/n!) times the integer coefficient of xᵏ, and k!/n! is an integer.
(ii) Since a − bx = b(π − x), f(x) = f(π − x); differentiating k times, f^(k)(π) = (−1)ᵏ f^(k)(0), an integer. Hence F(0) and F(π) are integers.
(iii) Because f^(2n+2) = 0, F'' + F = f. Therefore d/dx [F'(x) sin x − F(x) cos x] = (F''(x) + F(x)) sin x = f(x) sin x, and

    ∫₀^π f(x) sin x dx = [F'(x) sin x − F(x) cos x]₀^π = F(π) + F(0),  an integer.

(iv) For 0 < x < π we have x > 0, a − bx = b(π − x) > 0 and sin x > 0, so the integral is positive; and since x < π and a − bx < a, f(x) < πⁿ aⁿ / n!, so the integral is less than π (πa)ⁿ / n!, which tends to 0 as n grows. For n large enough the integral is an integer strictly between 0 and 1. Contradiction. ∎

**Corollary 3.1.** Every bracket [l, u] of π with rational endpoints, in particular every rational-endpoint bracket a computer produces, satisfies l < π < u strictly. So at every finite stage of any rational refinement the both band is nonempty: by Theorem 2 it can be made as small as desired, by Theorem 3 it is never empty. This is the exact content of the owner's sentence that π is the maybe that never closes. (Lindemann 1882 proved more: π is transcendental, so π is not algebraic either; cited, not proven here.)

## 4. Theorem 4 (π inside the law of accumulated uncertainty)

**Theorem 4.** ∫_{−∞}^{∞} e^{−x²} dx = √π.

Proof. Let I denote the integral; it converges and I > 0. By Fubini,
I² = ∫∫_{ℝ²} e^{−(x²+y²)} dx dy. In polar coordinates x = r cos θ, y = r sin θ, dx dy = r dr dθ,

    I² = ∫₀^{2π} ∫₀^{∞} e^{−r²} r dr dθ = 2π · [−e^{−r²}/2]₀^{∞} = 2π · ½ = π,

so I = √π. ∎

**Consequence, as a bounded example.** The normal density (2πσ²)^{−1/2} e^{−x²/(2σ²)} integrates to 1 because of Theorem 4. By the central limit theorem of Lindeberg and Lévy (cited, not proven here): for independent, identically distributed terms with finite positive variance, the centred and scaled sum converges in distribution to the standard normal law. Independence alone gives neither isotropy nor a Gaussian; the identical distribution, the finite variance and the centring and scaling are needed. Where π enters the computation is the line ∫₀^{2π} dθ, a full turn in the polar step of the proof.

## 5. What is proven, and what is not

Proven here: π is determined by the unending refinement of both cells, as their common limit with a certificate at every stage (Theorems 1 and 2); at every finite rational stage the both band is nonempty (Theorem 3, Corollary 3.1); the both cells are bound together by their shared source and are therefore few (Theorem 2); π is the normalizing constant of the Gaussian law (Theorem 4). Not implied: that refinement is the only way to determine π. Finite exact symbolic definitions exist (π = 4 arctan 1, Leibniz's series, Machin's formula); what the refinement adds is a determination with a certificate at every stage. Nor does irrationality say anything beyond rational endpoints; it constrains computed brackets, not thought in general.

Not proven, because not true as stated: "maybe = π" as an equation; "π is the equation of infinity". What those sentences point at is the Corollary: π is the standing example of a value every finite rational judgment leaves as both, and refinement shrinks that both without ever ending. The geometry demonstrates relational structure and approximation; it does not establish a universal law of thought, and the owner has asked that his sentences be read as a way of conveying one structure, not as separate claims.

## 6. Verification

- Theorem 1 and 2, numerically: the counts Y(n), M(n) match a brute-force cell-by-cell count for n ≤ 200 (Python, 13 Oct); the bracket contains π for n = 1, 2, 3, 10, 100, 1000; M(n) ≤ 3n holds for every n from 1 to 3000 with max M(n)/n = 2.000; Codex verified the same counts with exact integer arithmetic on 11 grids up to n = 15,848,932 (audit/bf16-harmony-20261013/ternary-sketch).
- Theorem 3, numerically (sanity only, a proof needs no numbers): with the hypothetical a/b = 22/7, the integral of (iii) is positive for n = 1, 2, 5, 10, 20, 30 and bounded as in (iv).
- Theorem 4: standard; any computer algebra system reproduces it.
- Open for Codex: independent reading of each step, especially the integer count in Theorem 2 and steps (i) to (iv) of Theorem 3.

## 7. Squaring the circle (added at the owner's words about Leonardo, 21:35 BST)

The grid of Theorems 1 and 2 is quadrature by refinement: square cells laid against a circle, the yes cells a square-made figure inside it, the yes and both cells a square-made figure around it, the both cells the remainder. The classical problem "square the circle" asks for one exact square with the circle's area, constructed with compass and straightedge.

**Theorem 5 (Lindemann 1882).** π is transcendental: it is the root of no polynomial with rational coefficients. Every length constructible with compass and straightedge is algebraic, a root of such a polynomial (a construction is a finite chain of linear and quadratic equations). A square with the area of the unit circle has side √π; if that side were constructible, √π and hence π would be algebraic. So the circle cannot be squared. (Lambert's irrationality, Theorem 3, is not enough for this, since some irrational numbers are constructible; Lindemann's transcendence is. Proof not reproduced here; a self-contained modern proof is Milla 2020, arXiv 2003.14035, after Hilbert.)

What this adds to Corollary 3.1: not only does every rational bracket leave the both band nonempty, no finite compass-and-straightedge construction reaches the circle's area exactly either. The remainder is permanent for those two kinds of finite means, rational brackets and classical constructions; other finite means do name π exactly (a symbolic definition, a convergent series), so the statement is about certificates and constructions, not about every finite means.

Leonardo da Vinci worked on exactly this, and that is documented: sheets of about 1505 in the Codex Atlanticus attempt to square the annulus (fol. 640r) and the lune (fol. 389r), and a notebook passage credits Archimedes with the first method and says Vitruvius came close without seeing it (Richter's edition, passage 1504). That the Vitruvian Man encodes a squaring of the circle is a reading by Irle and Schröer, who reach a ratio of 1.000373 between circle and square; Leonardo's manuscripts contain no statement about it, so it stays an interpretation. What is fact: he drew the human figure held in the circle and the square at once, and he spent years on the quadrature that Theorem 5 proves no finite construction can finish.
