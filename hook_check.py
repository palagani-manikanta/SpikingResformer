"""
hook_check.py -- does the patched neuron step (models/cbm.py: install_vmem_hook) change the network?

The CBM reads spikes through a patched multi_step_forward on the LIF node layers.2.6.down.0. This script runs
the SAME images through
  (a) the native SpikingResformer-Ti (SpikingJelly LIF, torch backend, no patch), recording that layer's output
      spikes with a plain forward hook, and
  (b) the patched backbone inside SpikingResformerCBM (spikes read from _hooked_lif._spike_seq),
and reports how many spikes differ and the largest difference in the backbone's ImageNet logits.
Nothing is trained; writes only hook_check/report.json and hook_check/report.md.

Usage (inside SpikingResformer-cbm):
  python hook_check.py                  # first 16 test images
  python hook_check.py --n 64
"""
import argparse, json, os, sys, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from spikingjelly.activation_based import functional

import demo_live as dl                      # same loaders, transform and CPU fallback as the demo

TAP = "layers.2.6.down.0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--indices-file", default=None)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    data = dl.load_test_cache()
    idx = (np.array(json.load(open(args.indices_file)), dtype=int) if args.indices_file
           else np.arange(args.n))

    native = dl.build_spiking_backbone().eval()
    cap = {}
    handle = dict(native.named_modules())[TAP].register_forward_hook(
        lambda m, i, o: cap.__setitem__("s", o.detach().clone()))
    cbm = dl.SpikingResformerCBM(backbone=dl.build_spiking_backbone(), n_concepts=112, n_classes=200,
                                 readout_type="spike_rate", backbone_dim=1536).to(dl.DEVICE).eval()
    native = native.to(dl.DEVICE)

    n_spikes, n_diff, max_logit_diff, n_pred_diff = 0, 0, 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idx), args.batch):
            b = idx[s:s + args.batch]
            x = torch.stack([dl.EVAL_TF(Image.open(os.path.join(dl.IMAGES_DIR, str(data["image_path"][i])))
                                        .convert("RGB")) for i in b]).to(dl.DEVICE)
            functional.reset_net(native)
            ln = native(x)
            functional.reset_net(cbm.backbone)
            lh = cbm.backbone(x)
            sn, sh = cap["s"], cbm._hooked_lif._spike_seq
            assert sn.shape == sh.shape, (sn.shape, sh.shape)
            n_spikes += sh.numel()
            n_diff += int((sn != sh).sum())
            max_logit_diff = max(max_logit_diff, float((ln - lh).abs().max()))
            n_pred_diff += int((ln.mean(0).argmax(-1) != lh.mean(0).argmax(-1)).sum())
    handle.remove()

    rep = {"layer": TAP, "n_images": int(len(idx)), "indices": idx.tolist(), "device": dl.DEVICE,
           "spike_values_compared": n_spikes, "spikes_differing": n_diff,
           "max_abs_backbone_logit_diff": max_logit_diff, "imagenet_predictions_differing": n_pred_diff,
           "identical": n_diff == 0 and max_logit_diff == 0.0}
    out = os.path.join(ROOT, "hook_check")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    md = (f"# Patched neuron step vs native SpikingJelly neuron ({TAP})\n\n"
          f"{len(idx)} test images, device {dl.DEVICE}.\n\n"
          f"- Spike values compared: {n_spikes:,}\n- Spikes differing: {n_diff}\n"
          f"- Largest difference in backbone ImageNet logits: {max_logit_diff:g}\n"
          f"- ImageNet predictions differing: {n_pred_diff}\n\n"
          f"Verdict: {'IDENTICAL' if rep['identical'] else 'DIFFERENT'}\n")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
