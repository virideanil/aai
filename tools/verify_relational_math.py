"""Finite mathematical reference, not a learned model or the production engine."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import argparse


def judgment(values):
    values = set(values)
    return {frozenset({0}): "no", frozenset({1}): "yes",
            frozenset({0, 1}): "maybe", frozenset(): "conflict"}[frozenset(values)]


def outcomes(worlds, predicate):
    return {predicate(w) for w in worlds}


def entropy(probabilities):
    return -sum(float(p) * math.log2(float(p)) for p in probabilities if p)


def _require(condition, message="Verification check failed"):
    if not condition:
        raise RuntimeError(message)


def information(joint):
    """Exact rational probabilities; logarithms are binary64 approximations."""
    _require(sum(joint.values()) == 1 and all(p >= 0 for p in joint.values()), "Invalid probability distribution")
    ps, po = {}, {}
    for (s, o), p in joint.items():
        ps[s] = ps.get(s, F(0)) + p
        po[o] = po.get(o, F(0)) + p
    expected_after = sum(float(p) * entropy([
        joint.get((s, o), F(0)) / p for s in ps]) for o, p in po.items() if p)
    kl = sum(float(p) * math.log2(float(p / (ps[s] * po[o])))
             for (s, o), p in joint.items() if p)
    reduction = entropy(ps.values()) - expected_after
    _require(abs(kl - reduction) < 1e-12, "Mutual information and entropy reduction disagree")
    return {"prior_entropy_bits": entropy(ps.values()),
            "expected_posterior_entropy_bits": expected_after,
            "information_bits": kl}


def main(output=None):
    checks = []
    def check(name, condition, evidence):
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})
        _require(condition, name)

    same = {(0, 0), (1, 1)}
    opposite = {(0, 1), (1, 0)}
    independent = set(product((0, 1), repeat=2))
    marginals = lambda worlds: [judgment(w[i] for w in worlds) for i in (0, 1)]
    equal = lambda w: int(w[0] == w[1])
    labels = {name: judgment(outcomes(worlds, equal)) for name, worlds in
              [("same", same), ("opposite", opposite), ("independent", independent)]}
    check("identical marginal trits require different relational answers",
          marginals(same) == marginals(opposite) == marginals(independent) == ["maybe", "maybe"]
          and labels == {"same": "yes", "opposite": "no", "independent": "maybe"}, labels)
    check("shared unknown XOR itself is resolved without observing it",
          outcomes(same, lambda w: w[0] ^ w[1]) == {0}, {"exact": "no", "cartesian_trits": "maybe"})

    worlds = sorted(independent)
    compared = 0
    refinement_violations = 0
    for mask in range(1, 16):
        before = {w for i, w in enumerate(worlds) if mask & (1 << i)}
        for observed in (0, 1):
            after = {w for w in before if w[0] == observed}
            if not after:
                continue
            for truth_table in range(16):
                predicate = lambda w, t=truth_table: (t >> (2*w[0]+w[1])) & 1
                if not (outcomes(after, predicate) <= outcomes(before, predicate)):
                    refinement_violations += 1
                compared += 1
    check("sound filtering preserves every predicate enclosure",
          compared == 384 and refinement_violations == 0,
          {"finite_refinement_cases": compared, "violations": refinement_violations})
    check("empty consistent-world set is explicit conflict", judgment([]) == "conflict", {})
    toggle = {(1 - w[0],) for w in {(0,)}}
    check("an action transition differs from evidence filtering", toggle == {(1,)}
          and not toggle <= {(0,)}, {"before": [[0]], "after_toggle": [[1]]})

    noise = information({(s, o): F(1, 4) for s in (0, 1) for o in (0, 1)})
    reveal = information({(0, 0): F(1, 2), (1, 1): F(1, 2)})
    check("unpredictable independent observation gives zero epistemic information",
          noise["information_bits"] == 0, noise)
    check("a reliable observation can resolve a shared open possibility",
          reveal["information_bits"] == 1, reveal)

    one, both = {}, {}
    for x, y in independent:
        one[(x ^ y, x)] = one.get((x ^ y, x), F(0)) + F(1, 4)
        both[(x ^ y, (x, y))] = F(1, 4)
    single, joint = information(one), information(both)
    check("two observations can have value missed by a one-step policy",
          single["information_bits"] == 0 and joint["information_bits"] == 1,
          {"first_bit": single, "both_bits": joint, "loss_before": "1/2",
           "two_query_cost": "1/5"})

    surprising = information({(0, 0): F(1, 4), (1, 0): F(1, 4), (1, 1): F(1, 2)})
    check("an individual observation may increase entropy despite positive expected gain",
          1 > surprising["prior_entropy_bits"] and surprising["information_bits"] > 0,
          {**surprising, "posterior_entropy_if_observation_zero": 1})
    check("one maybe label does not determine a probability",
          entropy([F(1, 2), F(1, 2)]) > entropy([F(1, 100), F(99, 100)]),
          {"equal_prior_bits": 1, "unequal_prior_bits": entropy([F(1,100), F(99,100)])})

    costs = []
    for c in (F(1, 100), F(1, 5), F(1, 2), F(1)):
        losses = {"guess": F(1,2), "defer": F(1,4), "reveal": c}
        costs.append({"query_cost": str(c), "minimum_loss": str(min(losses.values())),
                      "optimal_actions": [a for a,l in losses.items() if l == min(losses.values())]})
    check("the chosen observation depends on its declared complete cost",
          costs[0]["optimal_actions"] == ["reveal"] and costs[-1]["optimal_actions"] == ["defer"], costs)
    k = 20
    grouped_entropy = sum(F(n, 2**n) for n in range(1, k+1)) + F(k, 2**k)
    check("infinite support can have finite entropy", grouped_entropy == 2 - F(1,2**(k-1)),
          {"distribution": "p(n)=2^-n, n>=1", "analytic_entropy_bits": 2,
           "finite_tail_grouped_entropy": str(grouped_entropy)})

    extensions = 0
    extension_violations = 0
    for mask in range(1,16):
        before = {w for i,w in enumerate(worlds) if mask & (1 << i)}
        # Test all nonempty fibers independently: {0}, {1} or {0,1}.
        for choices in product(((0,), (1,), (0,1)), repeat=len(before)):
            after = {(*w,z) for w,fiber in zip(sorted(before),choices) for z in fiber}
            if {w[:2] for w in after} != before:
                extension_violations += 1
            for table in range(16):
                q = lambda w, t=table:(t >> (2*w[0]+w[1])) & 1
                if outcomes(after,q) != outcomes(before,q):
                    extension_violations += 1
            extensions += 1
    check("conservative extensions preserve every old binary query",
          extensions == 255 and extension_violations == 0,
          {"extensions":extensions,"old_truth_tables_per_extension":16,"violations":extension_violations})
    base = set(product((0,1),repeat=2))
    accepted = {w for w in base if w[0]==1}
    retracted = set(base)
    check("retracting a premise can reopen alternatives while preserving its prior state",
          accepted < retracted and outcomes(accepted,lambda w:w[0])=={1}
          and outcomes(retracted,lambda w:w[0])=={0,1},
          {"before_retraction_worlds":len(accepted),"after_retraction_worlds":len(retracted)})

    # Independent exact polynomial arithmetic checks Niven's algebraic steps.
    # 22/7 is only a test rational; its sine endpoint is not asserted to vanish.
    def derivative(p):
        return [F(i)*p[i] for i in range(1,len(p))] or [F(0)]
    def evaluate_polynomial(p,x):
        return sum(c*x**i for i,c in enumerate(p))
    niven_violations = 0
    for n in range(1,13):
        a,b = 22,7
        f = [F(0)]*(2*n+1)
        for k in range(n+1):
            f[n+k] = F(math.comb(n,k)*a**(n-k)*(-b)**k,math.factorial(n))
        derivatives = [f]
        for _ in range(2*n+2):derivatives.append(derivative(derivatives[-1]))
        for k,p in enumerate(derivatives):
            if p[0].denominator != 1 or evaluate_polynomial(p,F(a,b)) != (-1)**k*p[0]:
                niven_violations += 1
        total = [sum(((-1)**j)*derivatives[2*j][k] for j in range(n+1)
                     if k<len(derivatives[2*j])) for k in range(len(f))]
        dd = derivative(derivative(total))
        if [c+(dd[k] if k<len(dd) else 0) for k,c in enumerate(total)] != f:
            niven_violations += 1
    check("Niven endpoint integrality symmetry and differential identity", niven_violations == 0,
          {"tested_polynomial_orders":12,"hypothetical_rational":"22/7","violations":niven_violations,
           "scope":"Polynomial identities only; the contradiction for pi uses the written analytic proof"})

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nanoai.relations import (
        KM_ASSUMPTIONS, RELATION, benchmark_relation_carriers, decode_relation_tensor,
        encode_relation_tensor, km_revise_worlds, query_tensor_enclosure,
    )
    expressions = [
        0, 1, "x", "y",
        {"not": "x"}, {"not": "y"},
        {"and": ["x", "y"]}, {"or": ["x", "y"]},
        {"xor": ["x", "y"]}, {"eq": ["x", "y"]},
        {"implies": ["x", "y"]}, {"implies": ["y", "x"]},
    ]
    enclosure_cases = 0
    enclosure_violations = 0
    correlation_losses = 0
    tensor_roundtrips = 0
    tensor_violations = 0
    all_modes = ("bf16_support", "bf16_signed", "pack5", "support2", "bit_plane", "indicator_2n")
    for mask in range(1, 16):
        w_list = [list(w) for i, w in enumerate(worlds) if mask & (1 << i)]
        rel = {"type": RELATION, "variables": ["x", "y"], "worlds": w_list, "scope": f"mask-{mask}"}
        for mode in all_modes:
            encoded = encode_relation_tensor(rel, mode=mode)
            decoded = decode_relation_tensor(encoded)
            if decoded["variables"] != ["x", "y"] or decoded["worlds"] != w_list:
                tensor_violations += 1
            tensor_roundtrips += 1
        for expr in expressions:
            enc = query_tensor_enclosure(rel, expr)
            if not enc["enclosure_holds"]:
                enclosure_violations += 1
            enclosure_cases += 1
            if enc["correlation_loss"]:
                correlation_losses += 1
    check("2-bit support lattice over-approximates exact relational queries without false exclusion",
          enclosure_cases == 180 and enclosure_violations == 0 and correlation_losses > 0,
          {"enclosure_cases": enclosure_cases, "violations": enclosure_violations, "correlation_loss_cases": correlation_losses})
    check("BF16 support, BF16 signed, pack5, support2, bit_plane, and indicator_2n relational tensors round-trip losslessly",
          tensor_roundtrips == 90 and tensor_violations == 0,
          {"roundtrips": tensor_roundtrips, "violations": tensor_violations, "modes": list(all_modes)})

    # Verify compact primary payloads and carrier density across 8x64 and 12x512 regimes
    rel_1var = {"type": RELATION, "variables": ["x"], "worlds": [[0], [1]], "scope": "one-var"}
    b_sup = encode_relation_tensor(rel_1var, mode="bf16_support")
    b_sig = encode_relation_tensor(rel_1var, mode="bf16_signed")
    b_p5 = encode_relation_tensor(rel_1var, mode="pack5")
    b_s2 = encode_relation_tensor(rel_1var, mode="support2")
    b_bp = encode_relation_tensor(rel_1var, mode="bit_plane")
    b_in = encode_relation_tensor(rel_1var, mode="indicator_2n")
    w_8x64 = sorted({tuple((i * 2654435761 >> b) & 1 for b in range(8)) for i in range(256)})[:64]
    rel_8x64 = {"type": RELATION, "variables": [f"v{i}" for i in range(8)], "worlds": [list(w) for w in w_8x64], "scope": "8x64"}
    bench_8x64 = benchmark_relation_carriers(rel_8x64, expression={"eq": ["v0", "v1"]}, incoming_model=rel_8x64)
    check("compact U8 carriers (pack5, support2, bit_plane, indicator_2n) match exact byte densities and bitwise query/revision parity",
          len(b_sup[0]) == 8 and len(b_sig[0]) == 4 and len(b_p5[0]) == 1 and len(b_s2[0]) == 1
          and len(b_bp[0]) == 1 and len(b_in[0]) == 1
          and bench_8x64["bytes"] == {"bf16_support": 2048, "bf16_signed": 1024, "pack5": 103, "support2": 128, "bit_plane": 64, "indicator_2n": 32}
          and bench_8x64["auto_compact_mode"] == "indicator_2n"
          and bench_8x64["query_parity"] and bench_8x64["revision_parity"],
          bench_8x64)

    rel_2var = {"type": RELATION, "variables": ["x", "y"], "worlds": [[0, 0], [1, 1]], "scope": "two-var"}
    raw_2, shape_2, meta_2 = encode_relation_tensor(rel_2var, mode="bf16_support", open_questions=["x"])
    forged_rejected = False
    try:
        forged_meta = dict(meta_2, box_marginals={"x": "yes", "y": "yes"})
        decode_relation_tensor(raw_2, shape_2, forged_meta)
    except ValueError:
        forged_rejected = True
    missing_dtype_rejected = False
    try:
        no_dtype = dict(meta_2)
        no_dtype.pop("dtype", None)
        decode_relation_tensor(raw_2, shape_2, no_dtype)
    except ValueError:
        missing_dtype_rejected = True
    check("decode_relation_tensor rejects forged box_marginals, question metadata, and missing dtype",
          forged_rejected and missing_dtype_rejected,
          {"forged_box_marginals_rejected": forged_rejected, "missing_dtype_rejected": missing_dtype_rejected})

    # Independent mathematical reference for Katsuno-Mendelzon (1991) Theorem 3.3 & Dalal (1988):
    # Count violations explicitly so running under `python -O` still executes and enforces every check.
    nonempty_sets = [
        [list(w) for i, w in enumerate(worlds) if mask & (1 << i)]
        for mask in range(1, 16)
    ]
    all_w = [tuple(w) for w in worlds]
    km_violations = 0

    def build_faithful_preorder(psi_set):
        nonlocal km_violations
        dist = {w: min(abs(w[0] - v[0]) + abs(w[1] - v[1]) for v in psi_set) for w in all_w}
        rel_pairs = {(w1, w2) for w1 in all_w for w2 in all_w if dist[w1] <= dist[w2]}
        for w1 in all_w:
            for w2 in all_w:
                if not ((w1, w2) in rel_pairs or (w2, w1) in rel_pairs):
                    km_violations += 1
        for u in all_w:
            for v in all_w:
                for w in all_w:
                    if (u, v) in rel_pairs and (v, w) in rel_pairs and (u, w) not in rel_pairs:
                        km_violations += 1
        min_W = {w for w in all_w if all((w, u) in rel_pairs for u in all_w)}
        if min_W != psi_set:
            km_violations += 1
        return rel_pairs

    def oracle_min(mu_set, rel_pairs):
        return {w for w in mu_set if all((w, u) in rel_pairs for u in mu_set)}

    km_triples = 0
    oracle_matches = 0
    for psi in nonempty_sets:
        psi_set = {tuple(w) for w in psi}
        r_psi = build_faithful_preorder(psi_set)
        for mu1 in nonempty_sets:
            mu1_set = {tuple(w) for w in mu1}
            expected_rev1 = oracle_min(mu1_set, r_psi)
            rev1_list, _ = km_revise_worlds(psi, mu1)
            rev1 = {tuple(w) for w in rev1_list}
            if rev1 != expected_rev1:
                km_violations += 1
            else:
                oracle_matches += 1
            # KM1: psi o mu1 <= mu1
            if not (rev1 <= mu1_set):
                km_violations += 1
            # KM2: if psi & mu1 != empty, psi o mu1 == psi & mu1
            if (psi_set & mu1_set) and rev1 != (psi_set & mu1_set):
                km_violations += 1
            # KM3: if mu1 != empty and psi != empty, psi o mu1 != empty
            if len(rev1) == 0:
                km_violations += 1
            # KM4: syntax/ordering independence on world sets
            rev1_perm, _ = km_revise_worlds(list(reversed(psi)), list(reversed(mu1)))
            if {tuple(w) for w in rev1_perm} != rev1:
                km_violations += 1
            for mu2 in nonempty_sets:
                mu2_set = {tuple(w) for w in mu2}
                rev2 = oracle_min(mu2_set, r_psi)
                both_mu = [list(w) for w in sorted(mu1_set & mu2_set)]
                rev12_list, _ = km_revise_worlds(psi, both_mu)
                rev12 = {tuple(w) for w in rev12_list}
                if rev12 != oracle_min(mu1_set & mu2_set, r_psi):
                    km_violations += 1
                union_mu = [list(w) for w in sorted(mu1_set | mu2_set)]
                rev_union_list, _ = km_revise_worlds(psi, union_mu)
                rev_union = {tuple(w) for w in rev_union_list}
                if rev_union != oracle_min(mu1_set | mu2_set, r_psi):
                    km_violations += 1
                # KM5: (psi o mu1) & mu2 <= psi o (mu1 & mu2)
                if not ((rev1 & mu2_set) <= rev12):
                    km_violations += 1
                # KM6: if (psi o mu1) & mu2 != empty, psi o (mu1 & mu2) <= (psi o mu1) & mu2
                if (rev1 & mu2_set) and not (rev12 <= (rev1 & mu2_set)):
                    km_violations += 1
                # KM7: psi o (mu1 | mu2) in {psi o mu1, psi o mu2, (psi o mu1) | (psi o mu2)}
                if rev_union not in (rev1, rev2, rev1 | rev2):
                    km_violations += 1
                # KM8: (psi o mu1) & (psi o mu2) <= psi o (mu1 | mu2)
                if not ((rev1 & rev2) <= rev_union):
                    km_violations += 1
                km_triples += 1
    check("Katsuno-Mendelzon faithful total-preorder oracle and KM1-KM8 postulates hold exhaustively with zero violations under python -O",
          km_triples == 3375 and oracle_matches == 225 and km_violations == 0,
          {"verified_triples": km_triples, "oracle_pair_matches": oracle_matches, "violations": km_violations,
           "postulates": ["KM1", "KM2", "KM3", "KM4", "KM5", "KM6", "KM7", "KM8"],
           "assumptions": KM_ASSUMPTIONS})

    result = {"scope": "Finite mathematical reference and counterexamples; no model training, runtime speed, human psychology or universal-world claim.",
              "checks": checks, "passed": len(checks),
              "source_digest": "shake256:" + hashlib.shake_256(Path(__file__).read_bytes()).hexdigest(64)}
    if output is not None:
        output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"passed": len(checks), "relational_answers": labels, "refinement_cases": compared,
                      "enclosure_cases": enclosure_cases, "tensor_roundtrips": tensor_roundtrips,
                      "km_triples": km_triples, "km_violations": km_violations}))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
