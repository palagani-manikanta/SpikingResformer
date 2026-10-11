"""
nec_oracle.py -- ceiling of the 112 human CUB attributes under VLG-CBM's NEC = 5 protocol.

Question: can ANY model whose concepts are the 112 CUB attributes beat random-112 at NEC = 5? Feed the sparse
layer the TRUE class-level attribute vectors (train_fit -> test), i.e. a perfect concept predictor, with the same
solver, path, pruning and standardisation as nec_sparse_head.py. If this ceiling is below random-112 (43.9-47.8%),
no concept loss can pass the go/no-go on this concept set; if it is far above, better concept prediction can.
Analysis only; writes nec_oracle/report.{md,json}.
"""
import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import numpy as np, torch
import nec_sparse_head as nsh

def main():
    z = {s: np.load(os.path.join(ROOT, "seeds", "cache", f"{s}.npz")) for s in ("train_fit", "held_out", "test")}
    A = {s: z[s]["attrs"].astype(np.float64) for s in z}
    y = {s: z[s]["cids"].astype(np.int64) for s in z}
    rng = np.random.default_rng(0)
    out = {}
    for name, X in (("true attributes (perfect concepts)", A),
                    ("true attributes + 10% label flips", {s: np.where(rng.random(A[s].shape) < 0.10, 1 - A[s], A[s]) for s in A})):
        r = nsh.nec_evaluate(X["train_fit"], y["train_fit"], {"held_out": (X["held_out"], y["held_out"]), "test": (X["test"], y["test"])},
                             200, targets=(5, 10), device="cpu", dtype=torch.float32)
        out[name] = {k: r["targets"][k]["test_acc"] for k in (5, 10)}
        print(name, out[name], flush=True)
    os.makedirs(os.path.join(ROOT, "nec_oracle"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "nec_oracle", "report.json"), "w"), indent=1)
    md = ["# NEC ceiling of the 112 CUB attributes", "", "| Concept input | NEC = 5 test acc | NEC = 10 test acc |", "|---|---|---|"]
    md += [f"| {k} | {v[5]:.2f} | {v[10]:.2f} |" for k, v in out.items()]
    md += ["", "Reference at NEC = 5: trained SNN concepts 42.52, random-112 43.91 (SNN), 47.78 (ResNet-34)."]
    open(os.path.join(ROOT, "nec_oracle", "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()
