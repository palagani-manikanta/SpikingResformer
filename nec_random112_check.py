"""
nec_random112_check.py -- does the CPU set-up of nec_random112.py reproduce a published REAL unit?

Re-runs the trained-concept (real) unit of each main model, seed 0, at NEC = 5 with exactly the settings
nec_random112.py uses (CPU, float32, tolerance 1e-5, path stopped after NEC > 6) and compares test ANEC-5
with nec_sparse/units/<model>_seed0.json. If they agree, the random-112 numbers are comparable to the
published real and random-512 numbers. Writes only nec_random112/check_real_seed0.json.
"""
import json, os, sys, warnings
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")
import nec_random112 as nr
nsh = nr.nsh


def main():
    import calibration_all_models as cam
    import run_aug_views as rav
    cam.sanity = nr._cpu_sanity(cam, rav)
    nr._fast_path_and_dtype()
    feats = cam.Feats()
    n_concepts = rav.n_concepts_of(feats.rows["train_fit"])
    out = []
    for m in nsh.MAIN:
        u = nsh.run_unit(m, 0, "real", feats, n_concepts, targets=(5,), write=False)
        with open(os.path.join(nr.REAL_UNITS, f"{m}_seed0.json"), encoding="utf-8") as f:
            pub = json.load(f)["targets"]["5"]["test_acc"]
        new = u["targets"]["5"]["test_acc"]
        out.append({"model": m, "published_anec5": pub, "cpu_rerun_anec5": new, "diff": new - pub})
        print(f"{m}: published {pub:.2f}  cpu re-run {new:.2f}  diff {new - pub:+.2f}")
    nsh._write_json(out, os.path.join(nr.OUT_ROOT, "check_real_seed0.json"))


if __name__ == "__main__":
    main()
