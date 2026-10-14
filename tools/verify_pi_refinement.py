"""Current source arithmetic check; mathematical proof remains separately written."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import argparse
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
source = ROOT / "examples/trit-computer.html"
raw = source.read_bytes()
lines = raw.decode().splitlines()
names = ("isqrt", "counts", "ratDigits", "settled")
functions = []
for name in names:
    matches = [line.strip() for line in lines if line.strip().startswith(f"function {name}(")]
    assert len(matches) == 1, name
    functions.append(matches[0])
grids = sorted(set(range(1, 65)) | {2**k for k in range(7, 17)} | {10_000_000})
js = "\n".join(functions) + "\n" + (
    "process.stdout.write(JSON.stringify(" + json.dumps(grids)
    + ".map(n=>{const c=counts(n);return {n,...c,prefix:settled(c.Y,c.M,n)};})));"
)
started = time.perf_counter()
completed = subprocess.run(["node", "-e", js], check=True, capture_output=True, text=True)
observed = json.loads(completed.stdout)


def arctan_bounds(x, terms=24):
    partial = sum(((-1)**k * x**(2*k+1) / (2*k+1) for k in range(terms)), F(0))
    next_partial = partial + (-1)**terms * x**(2*terms+1) / (2*terms+1)
    return min(partial, next_partial), max(partial, next_partial)


# tan(4 arctan(1/5) - arctan(1/239)) = 1; the angle lies in (0, pi/2).
t2 = 2*F(1, 5)/(1-F(1, 5)**2)
t4 = 2*t2/(1-t2*t2)
assert (t4-F(1, 239))/(1+t4*F(1, 239)) == 1
a, b = arctan_bounds(F(1, 5)), arctan_bounds(F(1, 239))
pi_lo, pi_hi = 16*a[0]-4*b[1], 16*a[1]-4*b[0]
assert pi_lo < pi_hi

results = []
dyadic = []
for row in observed:
    n = row["n"]
    # Count by columns using arbitrary-precision integers, independent of JS rows.
    inside = sum(math.isqrt(n*n-(i+1)*(i+1)) for i in range(n))
    nonoutside = sum(math.isqrt(n*n-i*i-1)+1 for i in range(n))
    boundary, outside = nonoutside-inside, n*n-nonoutside
    assert (inside, boundary, outside) == (row["Y"], row["M"], row["N"])
    if n <= 64:
        brute = [0, 0, 0]
        for i in range(n):
            for j in range(n):
                category = (0 if (i+1)**2+(j+1)**2 <= n*n
                            else 2 if i*i+j*j >= n*n else 1)
                brute[category] += 1
        assert brute == [inside, boundary, outside]
    lo, hi = F(4*inside, n*n), F(4*(inside+boundary), n*n)
    assert lo < pi_lo < pi_hi < hi
    assert 0 < boundary <= 3*n and hi-lo <= F(12, n)
    prefix = "none yet"
    for digits in range(13):
        scale = 10**digits
        if (lo*scale).__floor__() != (hi*scale).__floor__():
            break
        q = (lo*scale).__floor__()
        rendered = str(q).rjust(digits+1, "0")
        prefix = rendered if not digits else rendered[:-digits]+"."+rendered[-digits:]
    assert row["prefix"] == prefix
    if n & (n-1) == 0:
        if dyadic:
            assert dyadic[-1][1] <= lo < hi <= dyadic[-1][2]
        dyadic.append((n, lo, hi))
    results.append({**row, "lower": str(lo), "upper": str(hi), "width": str(hi-lo), "passed": True})

result = {
    "source": "examples/trit-computer.html",
    "source_digest": "shake256:"+hashlib.shake_256(raw).hexdigest(64),
    "verified_functions": list(names), "grids": len(results),
    "brute_force_grids": 64, "dyadic_nested_stages": len(dyadic),
    "independent_reference": "Exact rational Machin formula with alternating-series enclosures",
    "reference": [str(pi_lo), str(pi_hi)],
    "cases": results, "seconds": time.perf_counter()-started,
    "scope": "Current JavaScript pure arithmetic and Python integer/rational references. No browser interaction, model learning or whole-task speed claim."
}
if args.output is not None:
    args.output.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k: result[k] for k in ("grids", "brute_force_grids", "dyadic_nested_stages", "seconds")}))
print(json.dumps(results[-1]))
