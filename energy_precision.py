"""
energy_precision.py -- one precision for compute AND memory (review item: "energy accounting is inconsistent on
precision").

WHY: energy_audit_v2 charges compute at 32-bit FLOAT costs (4.6 pJ/MAC = 3.7 mult + 0.9 add; 0.9 pJ/AC), while
energy_memory charges memory with 8-bit weights/activations. An 8-bit model would also use 8-bit arithmetic, which
is ~20-30x cheaper per operation, so memory then dominates and the compute ratio itself changes (an AC loses
less than a MAC). This script recomputes every number with ONE precision on both sides.

Horowitz (ISSCC 2014, 45 nm) operation energies, the same table every earlier report uses:
  32-bit float: add 0.9, mult 3.7 pJ          -> MAC 4.6, AC 0.9            (current paper)
  8-bit int:    add 0.03, mult 0.2 pJ         -> MAC 0.23, AC 0.03          (INT8, 8-bit accumulate)
  8-bit int mult, 32-bit int add 0.1 pJ       -> MAC 0.30, AC 0.10          (INT8, 32-bit accumulate; the usual
                                                                              hardware choice, avoids overflow)
Memory: the existing 8-bit reference scenario bytes (energy_memory_report.json, A8/W8, 16-bit membrane),
charged at 1 MB SRAM (12.5 pJ/B) and DRAM (162.5 / 325 pJ/B). For the FP32 row the 32-bit memory bytes would be
the consistent choice; that row is shown only to reproduce the published (inconsistent) numbers.

Op counts are read from energy_audit_v2_report.json (SNN stem once: AC count and MAC count; ANN MAC counts).
Analysis only; reads two JSON files; writes only energy_precision/report.{md,json}.
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "energy_precision")
A = json.load(open(os.path.join(ROOT, "energy_audit_v2", "energy_audit_v2_report.json"), encoding="utf-8"))
M = json.load(open(os.path.join(ROOT, "energy_memory", "energy_memory_report.json"), encoding="utf-8"))

E_MAC_FP32, E_AC_FP32 = A["constants"]["E_MAC_pJ"], A["constants"]["E_AC_pJ"]
snn = A["snn_stem_once_wrapper"]["truncated_at_tap"]
SNN_AC = snn["ac_ops_per_image"]
SNN_MAC = (snn["snn_mj_stem_once"] * 1e9 - SNN_AC * E_AC_FP32) / E_MAC_FP32       # stem once + GRU/CBL/head
ANN = {v["key"]: (v["label"], v["total_macs"]) for v in A["ann_variants"]}

PREC = {  # name: (pJ per MAC, pJ per AC, memory bit width)
    "Published: FP32 compute + 8-bit memory (mixed)": (4.6, 0.9, 8),
    "FP32 compute + 32-bit memory": (4.6, 0.9, 32),
    "INT8, 32-bit accumulate + 8-bit memory": (0.2 + 0.1, 0.1, 8),
    "INT8, 8-bit accumulate + 8-bit memory": (0.2 + 0.03, 0.03, 8),
}
MEM = {"compute only": 0.0, "1 MB SRAM (12.5 pJ/B)": 12.5, "DRAM low (162.5 pJ/B)": 162.5,
       "DRAM high (325 pJ/B)": 325.0}

GRID = lambda b: M["grid"][f"A{b}|dense|w_once|mem16|stem_once|lif_unfused"]["bytes"]["total"]   # reference scenario
snn_bytes = GRID(8)


rows, J = [], {"snn_ac": SNN_AC, "snn_mac": SNN_MAC, "snn_bytes_A8": snn_bytes, "ann": {}, "results": {}}
for pname, (emac, eac, bits) in PREC.items():
    snn_bytes = GRID(bits)
    ann_bytes = {k: M["anns"][str(bits)][k]["bytes"]["total"] for k in ANN}
    snn_c = (SNN_AC * eac + SNN_MAC * emac) * 1e-9                 # mJ
    for key in ("r34_mlp", "r18_mlp", "r50_linear"):
        label, macs = ANN[key]
        ann_c = macs * emac * 1e-9
        res = {}
        for mname, pjb in MEM.items():
            s = snn_c + snn_bytes * pjb * 1e-9
            a = ann_c + ann_bytes[key] * pjb * 1e-9
            res[mname] = {"snn_mj": s, "ann_mj": a, "ratio": a / s}
        be = (ann_c - snn_c) / ((snn_bytes - ann_bytes[key]) * 1e-9) if snn_bytes > ann_bytes[key] else float("inf")
        J["results"].setdefault(pname, {})[key] = {"snn_compute_mj": snn_c, "ann_compute_mj": ann_c,
                                                  "break_even_pJ_per_B": be, "by_memory": res}
        rows.append((pname, label, snn_c, ann_c, res, be))
        J["results"][pname][key]["snn_bytes"], J["results"][pname][key]["ann_bytes"] = snn_bytes, ann_bytes[key]

md = ["# Energy with one precision for compute and memory", "",
      f"SNN (stem once): {SNN_AC / 1e9:.3f} G AC + {SNN_MAC / 1e6:.1f} M MAC per image. Memory bytes: reference "
      f"scenario of energy_memory (16-bit membrane) at the stated width -- SNN {GRID(8) / 1e6:.1f} MB (8-bit) / "
      f"{GRID(32) / 1e6:.1f} MB (32-bit). ANN MACs from energy_audit_v2, bytes from energy_memory. Ratio = ANN energy / SNN energy (> 1: SNN cheaper). Break-even: memory pJ/B above which the "
      "SNN costs more.", "",
      "| Precision | ANN | SNN compute (mJ) | ANN compute (mJ) | " + " | ".join(f"Ratio, {m}" for m in MEM) +
      " | Break-even (pJ/B) |",
      "|---|---|---:|---:|" + "---:|" * len(MEM) + "---:|"]
for pname, label, sc, ac, res, be in rows:
    md.append(f"| {pname} | {label} | {sc:.3f} | {ac:.3f} | " +
              " | ".join(f"{res[m]['ratio']:.2f}x" for m in MEM) + f" | {be:.1f} |")
md += ["", "Notes:",
       "- The first row reproduces the published numbers (5.66x compute, 2.65x 1 MB SRAM, 0.45x/0.28x DRAM, "
       "break-even 55.1 pJ/B) and mixes 32-bit arithmetic with 8-bit memory.",
       "- Rows 2-4 are consistent: the arithmetic and the memory use the same precision.",
       "- Op energies: Horowitz ISSCC 2014, 45 nm. First-order model, no cache hierarchy; not a hardware measurement."]
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
json.dump(J, open(os.path.join(OUT, "report.json"), "w", encoding="utf-8"), indent=1)
print("\n".join(md))
