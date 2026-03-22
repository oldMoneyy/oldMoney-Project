#!/usr/bin/env python3
"""
Generation-Aware Calibration Data Builder for MiniCPM-SALA NVFP4 AWQ
=====================================================================

WHY THIS EXISTS:
  AWQ calibration runs forward-pass only (prefill). The MCQ eval failures
  are all reasoning loops during GENERATION (5k-40k tokens of math).
  AWQ never sees those activation patterns → can't protect those channels.

  This script collects FULL reasoning traces from the BF16 model:
    prompt + <think>...10k tokens of reasoning...</think>\nANSWER: X
  
  These traces are then used as calibration data so AWQ sees and
  protects the exact activation patterns that matter for convergence.

USAGE:
  # Step 1: Start your BF16 model on SGLang
  # Step 2: Run this script
  python collect_gen_traces.py \
      --api-base http://127.0.0.1:31333 \
      --model-name openbmb/MiniCPM-SALA \
      --existing-calib /opt/oldMoney-Project/quantization/calibration/final_merged_calibration.jsonl \
      --output /opt/oldMoney-Project/quantization/calibration/gen_aware_calib.jsonl
"""

import os
import sys
import json
import time
import random
import argparse
import requests
from pathlib import Path

# ============================================================
# STEM MCQ PROMPTS (physics-heavy, matching eval distribution)
# These are designed to trigger 5k-30k token reasoning chains
# ============================================================

STEM_MCQ_PROMPTS = [
    # --- QUANTUM MECHANICS (matches eval samples 12, 19, 22) ---
    """A spin-1/2 particle is in the state |ψ⟩ = (3|↑⟩ + 4i|↓⟩)/5. What is the expectation value of S_y?

A) ħ/2
B) -12ħ/25
C) 12ħ/25
D) -ħ/2

Please think step by step and provide your answer in the format ANSWER: X""",

    """Consider a quantum harmonic oscillator in the state |ψ⟩ = (|0⟩ + √2|1⟩ + |2⟩)/2. What is the expectation value of the energy?

A) ħω
B) 3ħω/2
C) 2ħω
D) 5ħω/2

Please think step by step and provide your answer in the format ANSWER: X""",

    """An electron is confined to a one-dimensional box of length L = 0.1 nm. What is the minimum energy of the electron in eV?

A) 3.8 eV
B) 15.1 eV
C) 37.7 eV
D) 150.8 eV

Please think step by step and provide your answer in the format ANSWER: X""",

    """A hydrogen atom is in the state ψ = (1/√3)|2,1,1⟩ + (1/√3)|2,1,0⟩ + (1/√3)|2,1,-1⟩. What is the expectation value of L_z?

A) 0
B) ħ/3
C) ħ
D) 2ħ/3

Please think step by step and provide your answer in the format ANSWER: X""",

    """For a particle in a 3D infinite square well of side L, the degeneracy of the energy level E = 14(π²ħ²)/(2mL²) is:

A) 1
B) 3
C) 6
D) 9

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- RELATIVITY & ASTROPHYSICS (matches eval samples 4, 14, 28) ---
    """A proton is accelerated from rest through a potential difference of 500 MV. What is its final kinetic energy in units of its rest mass energy (938.3 MeV)?

A) 0.53 m_p c²
B) 1.00 m_p c²
C) 500 m_p c²
D) The proton cannot reach this energy

Please think step by step and provide your answer in the format ANSWER: X""",

    """A star has an apparent magnitude of V = 12.0, a distance modulus of (m-M) = 8.0, and a color excess of E(B-V) = 0.5 mag. Assuming R_V = 3.1, what is the absolute magnitude M_V?

A) 2.45
B) 4.00
C) 5.55
D) 2.00

Please think step by step and provide your answer in the format ANSWER: X""",

    """A spacecraft in a circular orbit around Earth at radius r fires its engines to enter an elliptical transfer orbit with apoapsis at 4r. What is the ratio of the spacecraft's velocity just after the burn to the circular orbital velocity?

A) √(8/5)
B) √(6/5)
C) √(3/2)
D) √2

Please think step by step and provide your answer in the format ANSWER: X""",

    """Two events occur at the same location in frame S, separated by a time interval Δt = 3 μs. In frame S' moving at 0.8c relative to S, what is the spatial separation of these events?

A) 0 m
B) 720 m
C) 1200 m
D) 900 m

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- THERMODYNAMICS & STATISTICAL MECHANICS ---
    """An ideal Carnot engine operates between temperatures T_H = 600K and T_C = 300K, producing 1000 J of work per cycle. How much entropy is produced in the entire universe per cycle?

A) 0 J/K
B) 1.67 J/K
C) 3.33 J/K
D) 5.00 J/K

Please think step by step and provide your answer in the format ANSWER: X""",

    """Consider a system of N = 100 non-interacting spin-1/2 particles in a magnetic field B at temperature T. The partition function for a single spin is Z_1 = 2cosh(μB/kT). What is the average magnetization when μB/kT = 1?

A) 100μ·tanh(1)
B) 50μ·tanh(1)
C) 100μ·(e - e⁻¹)/(e + e⁻¹)
D) Both A and C are correct

Please think step by step and provide your answer in the format ANSWER: X""",

    """A monatomic ideal gas undergoes a process where PV² = constant. The molar heat capacity for this process is:

A) -R/2
B) R/2
C) 3R/2
D) 5R/2

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- ELECTROMAGNETISM ---
    """A conducting sphere of radius R carries charge Q. It is surrounded by a concentric conducting shell of inner radius 2R and outer radius 3R carrying charge -Q. What is the electric field at distance r = 2.5R from the center?

A) 0
B) kQ/r²
C) kQ/(2.5R)²
D) -kQ/(2.5R)²

Please think step by step and provide your answer in the format ANSWER: X""",

    """An electromagnetic wave in vacuum has an electric field amplitude E_0 = 100 V/m. What is the average energy density of this wave?

A) 4.43 × 10⁻⁸ J/m³
B) 8.85 × 10⁻⁸ J/m³
C) 2.22 × 10⁻⁸ J/m³
D) 1.77 × 10⁻⁷ J/m³

Please think step by step and provide your answer in the format ANSWER: X""",

    """A solenoid with 1000 turns per meter carries a current that increases linearly from 0 to 5A in 0.1 seconds. A circular loop of radius 3 cm is placed inside the solenoid perpendicular to the axis. What is the induced EMF in the loop?

A) 1.78 × 10⁻⁴ V
B) 1.78 × 10⁻⁵ V
C) 5.65 × 10⁻⁵ V
D) 5.65 × 10⁻⁴ V

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- CHEMISTRY ---
    """For the reaction 2NO₂(g) ⇌ N₂O₄(g), ΔH° = -57.2 kJ/mol and K_p = 6.7 at 298K. If the temperature is increased to 373K, how does K_p change? (Assume ΔH° is constant)

A) K_p increases to approximately 42
B) K_p decreases to approximately 0.7
C) K_p remains the same
D) K_p decreases to approximately 0.07

Please think step by step and provide your answer in the format ANSWER: X""",

    """A galvanic cell is constructed using Zn/Zn²⁺(0.01M) and Cu/Cu²⁺(1.0M) half-cells. Given E°(Zn²⁺/Zn) = -0.76V and E°(Cu²⁺/Cu) = +0.34V, what is the cell potential at 25°C?

A) 1.10 V
B) 1.16 V
C) 1.04 V
D) 0.52 V

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- BIOLOGY / BIOCHEMISTRY ---
    """In a dihybrid cross between two heterozygous parents (AaBb × AaBb), what fraction of the offspring will be homozygous for both recessive alleles (aabb)?

A) 1/4
B) 1/8
C) 1/16
D) 3/16

Please think step by step and provide your answer in the format ANSWER: X""",

    """A researcher performs a qPCR experiment and observes that the Ct value decreases by 3.32 cycles when the template concentration is increased 10-fold. What does this indicate about the amplification efficiency?

A) The efficiency is approximately 50%
B) The efficiency is approximately 75%
C) The efficiency is approximately 100%
D) The efficiency is approximately 200%

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- MATH / PROBABILITY ---
    """Let A be a 3×3 matrix with eigenvalues 1, 2, and 3. What is the trace of A⁻¹?

A) 6
B) 11/6
C) 1/6
D) 6/11

Please think step by step and provide your answer in the format ANSWER: X""",

    """A fair die is rolled repeatedly until a 6 appears. What is the expected number of rolls?

A) 3
B) 5
C) 6
D) 36

Please think step by step and provide your answer in the format ANSWER: X""",

    """The integral ∫₀^∞ x²e^(-x²)dx equals:

A) √π/4
B) √π/2
C) π/4
D) 1/2

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- COMPUTER SCIENCE ---
    """What is the time complexity of finding the shortest path between all pairs of vertices in a weighted directed graph with V vertices and E edges, using the Floyd-Warshall algorithm?

A) O(V²)
B) O(V² log V)
C) O(V³)
D) O(VE)

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- MORE PHYSICS (harder, longer reasoning required) ---
    """A uniform rod of length L and mass M is pivoted at one end and released from horizontal position. At the instant it passes through the vertical position, what is the angular velocity?

A) √(g/L)
B) √(2g/L)
C) √(3g/L)
D) √(6g/L)

Please think step by step and provide your answer in the format ANSWER: X""",

    """A satellite orbits Earth in an elliptical orbit with semi-major axis a = 26,600 km and eccentricity e = 0.74. Given that Earth's radius is 6,371 km and GM_Earth = 3.986 × 10¹⁴ m³/s², what is the orbital period?

A) 6 hours
B) 12 hours
C) 24 hours
D) 36 hours

Please think step by step and provide your answer in the format ANSWER: X""",

    """In Compton scattering, a photon of wavelength 0.05 nm scatters off a stationary electron at an angle of 90°. What is the wavelength of the scattered photon?

A) 0.0476 nm
B) 0.0524 nm
C) 0.0743 nm
D) 0.1000 nm

Please think step by step and provide your answer in the format ANSWER: X""",

    """A charged particle moves in a uniform magnetic field B = 0.5 T with a speed of 10⁷ m/s. If the particle's trajectory has a radius of curvature of 0.2 m, and the particle is either a proton or an alpha particle, which is it?

A) Proton
B) Alpha particle (He-4 nucleus)
C) Could be either, depending on the charge state
D) Neither; the mass is inconsistent with both

Please think step by step and provide your answer in the format ANSWER: X""",

    """A thin film of oil (n = 1.40) floats on water (n = 1.33). White light is incident normally from air. If the minimum thickness for constructive interference of 560 nm light (in air) is observed, what is that thickness?

A) 100 nm
B) 200 nm
C) 280 nm
D) 400 nm

Please think step by step and provide your answer in the format ANSWER: X""",

    """The wave function of a free particle is given by ψ(x,t) = Ae^(i(kx-ωt)) where k = 5 × 10¹⁰ m⁻¹. What is the de Broglie wavelength of this particle?

A) 0.63 Å
B) 1.26 Å
C) 2.00 Å
D) 6.28 Å

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- ORGANIC CHEMISTRY ---
    """In the bromination of toluene (methylbenzene) using Br₂/FeBr₃, the major product is:

A) Benzyl bromide (bromomethylbenzene)
B) ortho-bromotoluene and para-bromotoluene
C) meta-bromotoluene
D) 2,4,6-tribromoanisole

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- NUCLEAR PHYSICS ---
    """The binding energy per nucleon for Fe-56 is approximately 8.8 MeV. For U-235, it is approximately 7.6 MeV. If U-235 undergoes fission into two equal fragments, approximately how much energy is released per fission?

A) ~50 MeV
B) ~100 MeV
C) ~200 MeV
D) ~500 MeV

Please think step by step and provide your answer in the format ANSWER: X""",

    """A radioactive sample has a half-life of 5 days. After 20 days, what fraction of the original sample remains?

A) 1/4
B) 1/8
C) 1/16
D) 1/32

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- SOLID STATE / CONDENSED MATTER ---
    """In a simple cubic crystal with lattice constant a = 3 Å, what is the spacing between (110) planes?

A) 3.00 Å
B) 2.12 Å
C) 1.73 Å
D) 1.50 Å

Please think step by step and provide your answer in the format ANSWER: X""",

    # --- MECHANICS ---
    """Two blocks of masses m₁ = 2 kg and m₂ = 3 kg are connected by a light string over a frictionless pulley (Atwood machine). What is the acceleration of the system?

A) 1.96 m/s²
B) 3.92 m/s²
C) 4.90 m/s²
D) 9.80 m/s²

Please think step by step and provide your answer in the format ANSWER: X""",

    """A bullet of mass 10g traveling at 400 m/s embeds itself in a wooden block of mass 2 kg suspended as a pendulum. To what maximum height does the pendulum swing?

A) 0.02 m
B) 0.20 m
C) 0.40 m
D) 2.00 m

Please think step by step and provide your answer in the format ANSWER: X""",
]

# ============================================================
# GENERAL REASONING PROMPTS (non-MCQ, for diversity)
# These trigger different but still important reasoning patterns
# ============================================================

GENERAL_REASONING_PROMPTS = [
    "Derive the Euler-Lagrange equation from the principle of least action. Show all steps clearly.",

    "Explain why the eigenvalues of a Hermitian matrix are always real. Provide a rigorous proof.",

    "A researcher is designing a drug trial with 1000 participants. The expected effect size is 0.3 (Cohen's d), and they want 80% power at α = 0.05. Using a two-sample t-test framework, is this sample size sufficient? Show the calculation.",

    "Prove that the set of all 2×2 invertible matrices with real entries forms a group under matrix multiplication. Verify all four group axioms.",

    "Derive the blackbody radiation spectrum (Planck's law) starting from the quantization of energy in a cavity. Explain where classical physics fails (ultraviolet catastrophe) and how quantization resolves it.",
]


def check_server(api_base: str) -> bool:
    """Check if the SGLang server is running."""
    try:
        r = requests.get(f"{api_base}/v1/models", timeout=5)
        return r.status_code == 200
    except:
        return False


def generate_trace(api_base: str, model_name: str, prompt: str,
                   max_tokens: int = 32768, timeout: int = 600) -> dict:
    """Send prompt to BF16 model and get full reasoning trace."""
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Greedy for deterministic traces
    }

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        response = data["choices"][0]["message"]["content"]
        elapsed = time.time() - t0

        # Get token counts from usage if available
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return {
            "success": True,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_s": elapsed,
        }
    except requests.exceptions.Timeout:
        return {"success": False, "prompt": prompt, "error": "timeout"}
    except Exception as e:
        return {"success": False, "prompt": prompt, "error": str(e)}


def build_calibration_sequence(prompt: str, response: str) -> str:
    """
    Build the full calibration sequence.
    
    This concatenates prompt + response as a single text.
    When tokenized during calibration, the model's forward pass
    will see activation patterns for BOTH the prompt AND the
    full reasoning chain, including convergence and </think>.
    """
    # Simple concatenation — the calibration tokenizer will handle it.
    # The key is the CONTENT, not the chat template framing.
    return prompt.strip() + "\n\n" + response.strip()


def load_existing_long_context(path: str, max_samples: int = 30) -> list:
    """Load existing long-context calibration samples."""
    samples = []
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found, skipping existing data")
        return samples

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            text = data.get("question", "")
            # Keep only samples with substantial length (>5000 chars ≈ >1500 tokens)
            if len(text) > 5000:
                samples.append(data)

    if len(samples) > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Collect generation-aware calibration data from BF16 model"
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:31333",
                        help="SGLang API base URL")
    parser.add_argument("--model-name", default="openbmb/MiniCPM-SALA",
                        help="Model name for API calls")
    parser.add_argument("--existing-calib",
                        default="/opt/oldMoney-Project/quantization/calibration/final_merged_calibration.jsonl",
                        help="Path to existing calibration data (for long-context samples)")
    parser.add_argument("--output",
                        default="/opt/oldMoney-Project/quantization/calibration/gen_aware_calib.jsonl",
                        help="Output path for the final calibration dataset")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max tokens for generation (default: 32768)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per request in seconds (default: 600)")
    parser.add_argument("--max-mcq", type=int, default=25,
                        help="Max MCQ traces to collect (default: 25)")
    parser.add_argument("--max-reasoning", type=int, default=5,
                        help="Max general reasoning traces (default: 5)")
    parser.add_argument("--max-existing", type=int, default=30,
                        help="Max existing long-context samples to keep (default: 30)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Check server ──
    print(f"Checking server at {args.api_base}...")
    if not check_server(args.api_base):
        print(f"ERROR: Cannot reach SGLang server at {args.api_base}")
        print(f"Please start your BF16 model first, then re-run this script.")
        sys.exit(1)
    print(f"  Server is running ✓\n")

    # ── Phase 1: Collect MCQ reasoning traces ──
    print("=" * 70)
    print("PHASE 1: Collecting MCQ reasoning traces from BF16 model")
    print("  (This is the critical data that protects reasoning convergence)")
    print("=" * 70)

    mcq_traces = []
    prompts_to_use = STEM_MCQ_PROMPTS[:args.max_mcq + 10]  # extra buffer for failures
    random.seed(42)
    random.shuffle(prompts_to_use)

    for i, prompt in enumerate(prompts_to_use):
        if len(mcq_traces) >= args.max_mcq:
            break

        short_desc = prompt[:60].replace('\n', ' ')
        print(f"\n  [{i+1}/{len(prompts_to_use)}] Generating on: {short_desc}...")

        result = generate_trace(
            args.api_base, args.model_name, prompt,
            max_tokens=args.max_tokens, timeout=args.timeout
        )

        if not result["success"]:
            print(f"    FAILED: {result['error']}")
            continue

        resp = result["response"]
        resp_len = len(resp)
        has_think = "</think>" in resp
        has_answer = "ANSWER:" in resp.upper()

        # Only keep traces that show real reasoning (>1000 chars)
        if resp_len < 1000:
            print(f"    SKIPPED: too short ({resp_len} chars)")
            continue

        mcq_traces.append(result)
        status = "✓" if (has_think and has_answer) else "~"
        print(f"    {status} {resp_len:>6} chars, {result['completion_tokens']:>5} tokens, "
              f"</think>={'Y' if has_think else 'N'} ANSWER={'Y' if has_answer else 'N'} "
              f"({result['elapsed_s']:.1f}s)")

    print(f"\n  Collected {len(mcq_traces)} MCQ reasoning traces")

    # ── Phase 2: Collect general reasoning traces ──
    print(f"\n{'=' * 70}")
    print("PHASE 2: Collecting general reasoning traces")
    print("=" * 70)

    gen_reasoning_traces = []
    for i, prompt in enumerate(GENERAL_REASONING_PROMPTS[:args.max_reasoning]):
        short_desc = prompt[:60].replace('\n', ' ')
        print(f"\n  [{i+1}/{args.max_reasoning}] Generating on: {short_desc}...")

        result = generate_trace(
            args.api_base, args.model_name, prompt,
            max_tokens=args.max_tokens, timeout=args.timeout
        )

        if not result["success"]:
            print(f"    FAILED: {result['error']}")
            continue

        resp_len = len(result["response"])
        if resp_len < 500:
            print(f"    SKIPPED: too short ({resp_len} chars)")
            continue

        gen_reasoning_traces.append(result)
        print(f"    ✓ {resp_len:>6} chars, {result['completion_tokens']:>5} tokens "
              f"({result['elapsed_s']:.1f}s)")

    print(f"\n  Collected {len(gen_reasoning_traces)} general reasoning traces")

    # ── Phase 3: Load existing long-context data ──
    print(f"\n{'=' * 70}")
    print("PHASE 3: Loading existing long-context calibration data")
    print("=" * 70)

    existing_samples = load_existing_long_context(args.existing_calib, args.max_existing)
    print(f"  Loaded {len(existing_samples)} long-context samples")

    # ── Phase 4: Assemble final calibration dataset ──
    print(f"\n{'=' * 70}")
    print("PHASE 4: Assembling final calibration dataset")
    print("=" * 70)

    final_dataset = []

    # Add MCQ gen-traces (most important!)
    for trace in mcq_traces:
        seq = build_calibration_sequence(trace["prompt"], trace["response"])
        final_dataset.append({
            "question": seq,
            "_type": "mcq_gen_trace",
            "_tokens": trace["completion_tokens"],
        })

    # Add general reasoning traces
    for trace in gen_reasoning_traces:
        seq = build_calibration_sequence(trace["prompt"], trace["response"])
        final_dataset.append({
            "question": seq,
            "_type": "reasoning_gen_trace",
            "_tokens": trace["completion_tokens"],
        })

    # Add existing long-context samples
    for sample in existing_samples:
        sample["_type"] = "long_context_existing"
        final_dataset.append(sample)

    # Shuffle for balanced processing
    random.seed(42)
    random.shuffle(final_dataset)

    # ── Save ──
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ── Report ──
    print(f"\n{'=' * 70}")
    print("FINAL CALIBRATION DATASET SUMMARY")
    print(f"{'=' * 70}")

    type_counts = {}
    type_chars = {}
    for item in final_dataset:
        t = item.get("_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        type_chars[t] = type_chars.get(t, 0) + len(item["question"])

    total = len(final_dataset)
    print(f"\n  {'Category':<25} {'Count':>6} {'Avg chars':>12} {'% (per-sample)':>15}")
    print(f"  {'-' * 60}")
    for t in sorted(type_counts.keys()):
        avg_chars = type_chars[t] / type_counts[t]
        pct = type_counts[t] / total * 100
        print(f"  {t:<25} {type_counts[t]:>6} {avg_chars:>12,.0f} {pct:>14.1f}%")
    print(f"  {'TOTAL':<25} {total:>6}")

    print(f"""
  WITH PER-SAMPLE OBSERVER FIX:
    Each sample contributes 1/{total} = {1/total*100:.1f}% of Hessian
    MCQ gen-traces get {type_counts.get('mcq_gen_trace', 0)}/{total} = {type_counts.get('mcq_gen_trace', 0)/total*100:.0f}% ← protects reasoning convergence
    Long-context gets {type_counts.get('long_context_existing', 0)}/{total} = {type_counts.get('long_context_existing', 0)/total*100:.0f}% ← protects retrieval tasks
    
  WITHOUT OBSERVER FIX (token-weighted):
    MCQ gen-traces: ~{type_chars.get('mcq_gen_trace', 0) / sum(type_chars.values()) * 100:.0f}% by tokens
    Long-context: ~{type_chars.get('long_context_existing', 0) / sum(type_chars.values()) * 100:.0f}% by tokens
    ⚠ Long-context still dominates! Observer fix is ESSENTIAL.
""")

    print(f"  Saved to: {args.output}")
    print(f"\n  NEXT STEPS:")
    print(f"    1. Apply observer fix to nvfp4_awq.py (see patch_observer.py)")
    print(f"    2. Run quantization:")
    print(f"       python /opt/oldMoney-Project/quantization/nvfp4_awq.py \\")
    print(f"           --input /opt/model \\")
    print(f"           --output /opt/model_nvfp4_awq_v2 \\")
    print(f"           --calib-data {args.output} \\")
    print(f"           --max-samples {total} \\")
    print(f"           --max-len 131072 \\")
    print(f"           --mse-iters 200 \\")
    print(f"           --mse-max-shrink 0.60 \\")
    print(f"           --mse-error-norm 2.0")


if __name__ == "__main__":
    main()