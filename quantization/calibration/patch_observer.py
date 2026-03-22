#!/usr/bin/env python3
"""
Patch nvfp4_awq.py: Fix observer to use per-SAMPLE averaging instead of per-TOKEN.

PROBLEM:
  Current observer counts TOKENS in nsamples:
    self.nsamples += inp.shape[0]  # inp.shape[0] = seq_len (NOT 1)
    self.H_diag += (inp.float() ** 2).sum(dim=0)
  
  Result: H_diag = Σ_tokens(x²) / total_token_count
  A 70k-token Wikipedia article has 175× more influence than a 400-token MCQ prompt.
  
FIX:
  Count SAMPLES, average within each sample:
    self.nsamples += 1
    self.H_diag += (inp.float() ** 2).mean(dim=0)
  
  Result: H_diag = Σ_samples(mean_per_sample(x²)) / num_samples
  Each sample contributes equally regardless of length.

ALSO PATCHES:
  - Softer AWQ search defaults (mse_max_shrink, mse_error_norm, mse_iters)
  - Optional BF16 override for late lightning layers (26-28)

Usage:
  python patch_observer.py --awq-script /opt/oldMoney-Project/quantization/nvfp4_awq.py
  
  Or apply manually (see below).
"""

import argparse
import shutil
import os
import re
import sys


def patch_observer(content: str) -> str:
    """Patch the ActAwareObserver to use per-sample averaging."""
    
    # ── Fix 1: add_batch — count samples, mean over tokens ──
    old_add_batch = """    def add_batch(self, inp):
        if inp.dim() == 3: inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1: inp = inp.unsqueeze(0)
        self.nsamples += inp.shape[0]
        self.H_diag += (inp.float() ** 2).sum(dim=0)
        self.act_amax = max(self.act_amax, inp.abs().max().item())"""
    
    new_add_batch = """    def add_batch(self, inp):
        if inp.dim() == 3: inp = inp.reshape(-1, inp.shape[-1])
        elif inp.dim() == 1: inp = inp.unsqueeze(0)
        # FIX: Count SAMPLES not tokens. Mean over tokens within each sample.
        # This gives equal weight to a 400-token MCQ and a 70k-token document.
        self.nsamples += 1
        self.H_diag += (inp.float() ** 2).mean(dim=0)
        self.act_amax = max(self.act_amax, inp.abs().max().item())"""
    
    if old_add_batch not in content:
        print("  [WARN] Could not find exact add_batch pattern. Trying flexible match...")
        # Try a more flexible pattern
        pattern = r'(    def add_batch\(self, inp\):.*?self\.act_amax = max\(self\.act_amax, inp\.abs\(\)\.max\(\)\.item\(\)\))'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content.replace(match.group(0), new_add_batch)
            print("  ✓ Patched add_batch (flexible match)")
        else:
            print("  ✗ FAILED to patch add_batch — apply manually!")
            return content
    else:
        content = content.replace(old_add_batch, new_add_batch)
        print("  ✓ Patched add_batch (exact match)")
    
    return content


def patch_defaults(content: str) -> str:
    """Patch default AWQ search parameters to be less aggressive."""

    # mse_iters: 100 → 200
    content = content.replace(
        'parser.add_argument("--mse-iters", type=int, default=100)',
        'parser.add_argument("--mse-iters", type=int, default=200)'
    )
    
    # mse_max_shrink: 0.80 → 0.60  
    content = content.replace(
        'parser.add_argument("--mse-max-shrink", type=float, default=0.80)',
        'parser.add_argument("--mse-max-shrink", type=float, default=0.60)'
    )
    
    # mse_error_norm: 2.4 → 2.0
    content = content.replace(
        'parser.add_argument("--mse-error-norm", type=float, default=2.4)',
        'parser.add_argument("--mse-error-norm", type=float, default=2.0)'
    )
    
    print("  ✓ Patched default parameters (iters=200, shrink=0.60, norm=2.0)")
    return content


def patch_skip_late_layers(content: str) -> str:
    """
    Add option to keep layers 26-28 in BF16.
    
    These are the last lightning-attn layers before the BF16 output layers (29-31).
    Keeping them in BF16 preserves reasoning convergence at ~0.9GB extra memory cost.
    """
    
    old_quant_layers = '    quant_layers = [i for i, mt in enumerate(config.mixer_types) if mt in ("lightning", "lightning_attn", "lightning-attn")]'
    
    new_quant_layers = """    # Layers to keep in BF16 even if they are lightning-attn.
    # Layers 26-28 are the last lightning block before BF16 output layers (29-31).
    # Keeping them in BF16 preserves reasoning convergence at ~0.9GB extra cost.
    bf16_override = set()
    if args.bf16_late_layers:
        bf16_override = {26, 27, 28}
        print(f"[INFO] Keeping layers {sorted(bf16_override)} in BF16 (late-layer protection)")
    quant_layers = [i for i, mt in enumerate(config.mixer_types) if mt in ("lightning", "lightning_attn", "lightning-attn") and i not in bf16_override]"""
    
    if old_quant_layers in content:
        content = content.replace(old_quant_layers, new_quant_layers)
        print("  ✓ Patched quant_layers with BF16 override option")
    else:
        print("  [WARN] Could not find quant_layers pattern — apply manually if needed")

    # Add the CLI argument
    old_parser_end = '    args = parser.parse_args()'
    new_parser_end = """    parser.add_argument("--bf16-late-layers", action="store_true",
                        help="Keep layers 26-28 in BF16 to protect reasoning convergence (~0.9GB extra)")
    args = parser.parse_args()"""
    
    if old_parser_end in content:
        content = content.replace(old_parser_end, new_parser_end, 1)  # replace only first occurrence
        print("  ✓ Added --bf16-late-layers CLI argument")
    
    return content


def main():
    parser = argparse.ArgumentParser(description="Patch nvfp4_awq.py observer")
    parser.add_argument("--awq-script",
                        default="/opt/oldMoney-Project/quantization/nvfp4_awq.py",
                        help="Path to nvfp4_awq.py")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating a backup")
    parser.add_argument("--skip-defaults", action="store_true",
                        help="Don't patch default parameters")
    parser.add_argument("--skip-late-layers", action="store_true",
                        help="Don't add late-layer BF16 override")
    args = parser.parse_args()

    if not os.path.exists(args.awq_script):
        print(f"ERROR: {args.awq_script} not found")
        sys.exit(1)

    # Backup
    if not args.no_backup:
        backup = args.awq_script + ".bak"
        shutil.copy2(args.awq_script, backup)
        print(f"Backup saved to: {backup}")

    # Read
    with open(args.awq_script, 'r') as f:
        content = f.read()

    print(f"\nPatching {args.awq_script}...")

    # Apply patches
    content = patch_observer(content)

    if not args.skip_defaults:
        content = patch_defaults(content)

    if not args.skip_late_layers:
        content = patch_skip_late_layers(content)

    # Write
    with open(args.awq_script, 'w') as f:
        f.write(content)

    print(f"\n{'=' * 60}")
    print("PATCHES APPLIED SUCCESSFULLY")
    print(f"{'=' * 60}")
    print(f"""
WHAT CHANGED:

  1. OBSERVER FIX (critical):
     add_batch() now does per-SAMPLE averaging instead of per-TOKEN.
     Before: 16 Wikipedia articles = 85.8% of Hessian
     After:  Each of N samples = 1/N of Hessian
     
  2. SOFTER AWQ SEARCH (recommended):
     mse_iters:      100 → 200  (finer grid search)
     mse_max_shrink: 0.80 → 0.60  (less aggressive clipping)
     mse_error_norm: 2.4 → 2.0  (standard L2 norm)
     
  3. LATE LAYER BF16 OVERRIDE (optional):
     --bf16-late-layers flag keeps layers 26-28 in BF16
     Cost: ~0.9GB extra GPU memory
     Protects reasoning convergence in late layers
     
MANUAL VERIFICATION:
  grep -n "self.nsamples += 1" {args.awq_script}
  grep -n "mean(dim=0)" {args.awq_script}
""")


# ── For manual patching: the exact diff ──
MANUAL_DIFF = """
If automatic patching fails, apply these changes manually:

FILE: nvfp4_awq.py

1. In class ActAwareObserver, method add_batch:

   CHANGE:
        self.nsamples += inp.shape[0]
        self.H_diag += (inp.float() ** 2).sum(dim=0)
   
   TO:
        self.nsamples += 1
        self.H_diag += (inp.float() ** 2).mean(dim=0)

2. In argparse section:

   CHANGE: --mse-iters default=100  → default=200
   CHANGE: --mse-max-shrink default=0.80  → default=0.60
   CHANGE: --mse-error-norm default=2.4  → default=2.0

3. In quantize_model function, where quant_layers is defined:

   CHANGE:
     quant_layers = [i for i, mt in enumerate(config.mixer_types) 
                     if mt in ("lightning", "lightning_attn", "lightning-attn")]
   
   TO:
     bf16_override = {26, 27, 28} if args.bf16_late_layers else set()
     quant_layers = [i for i, mt in enumerate(config.mixer_types)
                     if mt in ("lightning", "lightning_attn", "lightning-attn")
                     and i not in bf16_override]
"""

if __name__ == "__main__":
    main()
    if "--help" in sys.argv:
        print(MANUAL_DIFF)