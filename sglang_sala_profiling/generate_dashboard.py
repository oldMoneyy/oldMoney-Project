#!/usr/bin/env python3
"""Generate interactive HTML profiling dashboard from SALA profiling results."""
import csv
import os
import re
import json
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
QUANT_HTML = os.path.join(SCRIPT_DIR, "quantization_summary.html")
OUTPUT_HTML = os.path.join(SCRIPT_DIR, "quantization_summary.html")

BACKENDS = ["flashinfer", "minicpm_flashinfer"]
TIERS = ["c1", "c8", "c64"]

# Layer type mapping
MINICPM4_LAYERS = {0, 9, 16, 17, 22, 29, 30, 31}
LIGHTNING_LAYERS = set(range(32)) - MINICPM4_LAYERS


def parse_bench_result(filepath):
    """Extract key metrics from bench_serving output."""
    metrics = {}
    candidates = [filepath]
    for suffix in ["bench_sala", "bench_phase1"]:
        candidates.append(filepath.replace("bench_baseline", suffix))

    text = ""
    for c in candidates:
        try:
            with open(c) as f:
                text = f.read()
            if "Serving Benchmark Result" in text:
                break
        except FileNotFoundError:
            continue

    if not text or "Serving Benchmark Result" not in text:
        return None

    patterns = {
        "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
        "input_throughput": r"Input token throughput \(tok/s\):\s+([0-9.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
        "total_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
        "median_itl_ms": r"Median ITL \(ms\):\s+([0-9.]+)",
        "p95_itl_ms": r"P95 ITL \(ms\):\s+([0-9.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
        "max_itl_ms": r"Max ITL \(ms\):\s+([0-9.]+)",
        "mean_e2e_ms": r"Mean E2E Latency \(ms\):\s+([0-9.]+)",
        "successful_requests": r"Successful requests:\s+(\d+)",
        "total_input_tokens": r"Total input tokens:\s+(\d+)",
        "total_output_tokens": r"Total generated tokens:\s+(\d+)",
        "peak_output_throughput": r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
        "concurrency": r"Concurrency:\s+([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics if metrics else None


def parse_sala_csv(filepath):
    """Parse SALA profiler CSV into structured data."""
    if not os.path.exists(filepath):
        return None
    rows = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "key": row["key"],
                "total_ms": float(row["total_ms"]),
                "count": int(row["count"]),
                "avg_ms": float(row["avg_ms"]),
                "min_ms": float(row["min_ms"]),
                "max_ms": float(row["max_ms"]),
            })
    return rows


def parse_gpu_analysis(filepath):
    """Parse GPU analysis text."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        text = f.read()
    info = {}
    for key, pat in [
        ("avg_sm_pct", r"Avg SM%:\s+([0-9.]+)%"),
        ("avg_mem_pct", r"Avg Mem%:\s+([0-9.]+)%"),
        ("max_sm_pct", r"Max SM%:\s+(\d+)%"),
        ("peak_mem_mib", r"Peak:\s+(\d+) MiB"),
        ("min_mem_mib", r"Min:\s+(\d+) MiB"),
        ("avg_mem_mib", r"Avg:\s+(\d+) MiB"),
    ]:
        m = re.search(pat, text)
        if m:
            info[key] = float(m.group(1))
    return info if info else None


def aggregate_sala_data(rows):
    """Aggregate SALA CSV rows into useful structures."""
    if not rows:
        return None

    # Per-layer totals
    layer_data = defaultdict(lambda: {"total_ms": 0, "prefill_ms": 0, "decode_ms": 0, "type": ""})
    # Component breakdowns
    component_data = defaultdict(lambda: {"prefill_ms": 0, "decode_ms": 0, "count_p": 0, "count_d": 0})
    # Per-layer component detail
    layer_component = defaultdict(lambda: defaultdict(lambda: {"prefill_ms": 0, "decode_ms": 0}))

    for row in rows:
        key = row["key"]
        parts = key.split("/")
        if len(parts) < 3:
            # model/embed, model/logits etc
            mode = parts[-1] if parts[-1] in ("prefill", "decode") else "other"
            comp_key = "/".join(parts[:-1]) if mode != "other" else key
            if mode == "prefill":
                component_data[f"non-layer/{comp_key}"]["prefill_ms"] += row["total_ms"]
            elif mode == "decode":
                component_data[f"non-layer/{comp_key}"]["decode_ms"] += row["total_ms"]
            continue

        layer_key = parts[0]  # L00, L01, ...
        layer_num = int(layer_key[1:]) if layer_key.startswith("L") else -1
        if layer_num < 0:
            continue

        ltype = "minicpm4" if layer_num in MINICPM4_LAYERS else "lightning"

        # Determine mode from last part
        mode = parts[-1] if parts[-1] in ("prefill", "decode") else "other"

        # Component name: everything between layer type and mode
        if len(parts) >= 3:
            # e.g. L00/minicpm4/attn/prefill -> comp = "attn"
            # e.g. L01/lightning/fla_kernel/decode -> comp = "fla_kernel"
            comp = parts[2] if len(parts) >= 4 else parts[1]

        layer_data[layer_num]["type"] = ltype
        if mode == "prefill":
            layer_data[layer_num]["prefill_ms"] += row["total_ms"]
            layer_data[layer_num]["total_ms"] += row["total_ms"]
            component_data[f"{ltype}/{comp}"]["prefill_ms"] += row["total_ms"]
            component_data[f"{ltype}/{comp}"]["count_p"] += row["count"]
            layer_component[layer_num][comp]["prefill_ms"] += row["total_ms"]
        elif mode == "decode":
            layer_data[layer_num]["decode_ms"] += row["total_ms"]
            layer_data[layer_num]["total_ms"] += row["total_ms"]
            component_data[f"{ltype}/{comp}"]["decode_ms"] += row["total_ms"]
            component_data[f"{ltype}/{comp}"]["count_d"] += row["count"]
            layer_component[layer_num][comp]["decode_ms"] += row["total_ms"]

    # Compute totals
    total_prefill = sum(d["prefill_ms"] for d in layer_data.values())
    total_decode = sum(d["decode_ms"] for d in layer_data.values())
    lightning_total = sum(d["total_ms"] for d in layer_data.values() if d["type"] == "lightning")
    minicpm4_total = sum(d["total_ms"] for d in layer_data.values() if d["type"] == "minicpm4")

    return {
        "layer_data": dict(layer_data),
        "component_data": dict(component_data),
        "layer_component": {k: dict(v) for k, v in layer_component.items()},
        "total_prefill_ms": total_prefill,
        "total_decode_ms": total_decode,
        "lightning_total_ms": lightning_total,
        "minicpm4_total_ms": minicpm4_total,
        "grand_total_ms": total_prefill + total_decode,
    }


def collect_all_data():
    """Collect all profiling data."""
    data = {"benchmarks": {}, "sala": {}, "gpu": {}}

    for backend in BACKENDS:
        for tier in TIERS:
            key = f"{backend}/{tier}"
            base_dir = os.path.join(RESULTS_DIR, backend, tier)

            # Benchmark
            bench_file = os.path.join(base_dir, "bench_baseline.txt")
            data["benchmarks"][key] = parse_bench_result(bench_file)

            # SALA CSV
            csv_file = os.path.join(base_dir, "sala_timings.csv")
            raw = parse_sala_csv(csv_file)
            data["sala"][key] = aggregate_sala_data(raw) if raw else None
            # Also keep raw for detailed view
            if raw:
                data["sala"][key]["raw"] = raw

            # GPU
            gpu_file = os.path.join(base_dir, "gpu_analysis.txt")
            data["gpu"][key] = parse_gpu_analysis(gpu_file)

    return data


def generate_html(data):
    """Generate the full HTML dashboard."""

    # Prepare JSON-safe data for embedding
    bench_json = {}
    for key, val in data["benchmarks"].items():
        if val:
            bench_json[key] = val

    sala_json = {}
    for key, val in data["sala"].items():
        if val:
            # Don't include raw in JSON (too large), just aggregates
            sala_json[key] = {k: v for k, v in val.items() if k != "raw"}

    gpu_json = {}
    for key, val in data["gpu"].items():
        if val:
            gpu_json[key] = val

    # Build per-layer timing tables for each config that has SALA data
    layer_tables_html = ""
    for config_key, sdata in data["sala"].items():
        if not sdata or "raw" not in sdata:
            continue
        raw = sdata["raw"]

        # Build layer-level rows
        ld = sdata["layer_data"]
        grand = sdata["grand_total_ms"]
        rows_html = ""
        for ln in sorted(ld.keys()):
            d = ld[ln]
            pct = (d["total_ms"] / grand * 100) if grand > 0 else 0
            ltype_cls = "minicpm4" if d["type"] == "minicpm4" else "lightning"
            rows_html += f"""<tr class="layer-row {ltype_cls}" data-layer="{ln}" data-config="{config_key}">
                <td>L{ln:02d}</td>
                <td><span class="badge badge-{ltype_cls}">{d['type']}</span></td>
                <td class="num">{d['total_ms']:.1f}</td>
                <td class="num">{d['prefill_ms']:.1f}</td>
                <td class="num">{d['decode_ms']:.1f}</td>
                <td class="num">{pct:.1f}%</td>
                <td><div class="bar-container"><div class="bar bar-{ltype_cls}" style="width:{min(pct*2, 100):.1f}%"></div></div></td>
            </tr>\n"""

        layer_tables_html += f"""
        <div class="layer-table-section" data-config="{config_key}">
            <h3>{config_key}</h3>
            <div class="summary-row">
                <span>Total: {grand:.0f} ms</span>
                <span>Prefill: {sdata['total_prefill_ms']:.0f} ms ({sdata['total_prefill_ms']/grand*100:.1f}%)</span>
                <span>Decode: {sdata['total_decode_ms']:.0f} ms ({sdata['total_decode_ms']/grand*100:.1f}%)</span>
                <span class="badge badge-lightning">Lightning: {sdata['lightning_total_ms']:.0f} ms ({sdata['lightning_total_ms']/grand*100:.1f}%)</span>
                <span class="badge badge-minicpm4">MiniCPM4: {sdata['minicpm4_total_ms']:.0f} ms ({sdata['minicpm4_total_ms']/grand*100:.1f}%)</span>
            </div>
            <table class="data-table">
                <thead><tr><th>Layer</th><th>Type</th><th>Total (ms)</th><th>Prefill (ms)</th><th>Decode (ms)</th><th>% of Total</th><th>Bar</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>"""

    # Build component breakdown tables
    component_tables_html = ""
    for config_key, sdata in data["sala"].items():
        if not sdata:
            continue
        cd = sdata["component_data"]
        grand = sdata["grand_total_ms"]

        # Separate by type
        for ltype in ["lightning", "minicpm4"]:
            type_comps = {k: v for k, v in cd.items() if k.startswith(ltype + "/")}
            if not type_comps:
                continue
            type_total = sum(v["prefill_ms"] + v["decode_ms"] for v in type_comps.values())
            rows = ""
            for comp_key in sorted(type_comps.keys(), key=lambda x: -(type_comps[x]["prefill_ms"] + type_comps[x]["decode_ms"])):
                v = type_comps[comp_key]
                comp_name = comp_key.split("/", 1)[1]
                total = v["prefill_ms"] + v["decode_ms"]
                pct = (total / type_total * 100) if type_total > 0 else 0
                rows += f"""<tr>
                    <td>{comp_name}</td>
                    <td class="num">{total:.1f}</td>
                    <td class="num">{v['prefill_ms']:.1f}</td>
                    <td class="num">{v['decode_ms']:.1f}</td>
                    <td class="num">{pct:.1f}%</td>
                    <td><div class="bar-container"><div class="bar bar-{ltype}" style="width:{min(pct, 100):.1f}%"></div></div></td>
                </tr>\n"""

            component_tables_html += f"""
            <div class="component-section" data-config="{config_key}" data-type="{ltype}">
                <h4>{config_key} - {ltype.upper()} Components (Total: {type_total:.0f} ms)</h4>
                <table class="data-table">
                    <thead><tr><th>Component</th><th>Total (ms)</th><th>Prefill (ms)</th><th>Decode (ms)</th><th>% of Type</th><th>Bar</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>"""

    # Build per-layer detailed component breakdown
    detail_tables_html = ""
    for config_key, sdata in data["sala"].items():
        if not sdata:
            continue
        lc = sdata["layer_component"]
        for ln in sorted(lc.keys()):
            comps = lc[ln]
            ltype = "minicpm4" if ln in MINICPM4_LAYERS else "lightning"
            layer_total = sum(v["prefill_ms"] + v["decode_ms"] for v in comps.values())
            rows = ""
            for comp in sorted(comps.keys(), key=lambda x: -(comps[x]["prefill_ms"] + comps[x]["decode_ms"])):
                v = comps[comp]
                total = v["prefill_ms"] + v["decode_ms"]
                pct = (total / layer_total * 100) if layer_total > 0 else 0
                rows += f"<tr><td>{comp}</td><td class='num'>{total:.1f}</td><td class='num'>{v['prefill_ms']:.1f}</td><td class='num'>{v['decode_ms']:.1f}</td><td class='num'>{pct:.1f}%</td></tr>\n"

            detail_tables_html += f"""
            <div class="detail-layer" data-config="{config_key}" data-layer="{ln}" style="display:none;">
                <h4>L{ln:02d} ({ltype}) - {config_key} - Total: {layer_total:.1f} ms</h4>
                <table class="data-table compact">
                    <thead><tr><th>Component</th><th>Total (ms)</th><th>Prefill (ms)</th><th>Decode (ms)</th><th>%</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>"""

    # Benchmark comparison table
    bench_rows = ""
    for backend in BACKENDS:
        for tier in TIERS:
            key = f"{backend}/{tier}"
            b = data["benchmarks"].get(key)
            g = data["gpu"].get(key)
            if not b:
                continue
            bench_rows += f"""<tr>
                <td><b>{backend}</b></td><td>{tier}</td>
                <td class="num">{b.get('total_throughput', 0):.1f}</td>
                <td class="num">{b.get('input_throughput', 0):.1f}</td>
                <td class="num">{b.get('output_throughput', 0):.1f}</td>
                <td class="num">{b.get('peak_output_throughput', 0):.0f}</td>
                <td class="num">{b.get('mean_ttft_ms', 0):.0f}</td>
                <td class="num">{b.get('median_ttft_ms', 0):.0f}</td>
                <td class="num">{b.get('p99_ttft_ms', 0):.0f}</td>
                <td class="num">{b.get('mean_tpot_ms', 0):.2f}</td>
                <td class="num">{b.get('median_tpot_ms', 0):.2f}</td>
                <td class="num">{b.get('mean_itl_ms', 0):.2f}</td>
                <td class="num">{b.get('median_itl_ms', 0):.2f}</td>
                <td class="num">{b.get('p99_itl_ms', 0):.2f}</td>
                <td class="num">{b.get('max_itl_ms', 0):.1f}</td>
                <td class="num">{b.get('duration_s', 0):.1f}</td>
                <td class="num">{b.get('concurrency', 0):.1f}</td>
                <td class="num">{f"{g.get('avg_sm_pct', 0):.0f}%" if g else 'N/A'}</td>
                <td class="num">{f"{g.get('peak_mem_mib', 0)/1024:.1f} GB" if g else 'N/A'}</td>
            </tr>\n"""

    # Compute speedup ratios
    speedup_html = ""
    for tier in TIERS:
        fi = data["benchmarks"].get(f"flashinfer/{tier}")
        mcfi = data["benchmarks"].get(f"minicpm_flashinfer/{tier}")
        if fi and mcfi:
            fi_out = fi.get("output_throughput", 1)
            mcfi_out = mcfi.get("output_throughput", 1)
            ratio = fi_out / mcfi_out if mcfi_out > 0 else 0
            fi_ttft = fi.get("mean_ttft_ms", 0)
            mcfi_ttft = mcfi.get("mean_ttft_ms", 0)
            ttft_ratio = mcfi_ttft / fi_ttft if fi_ttft > 0 else 0
            fi_tpot = fi.get("mean_tpot_ms", 0)
            mcfi_tpot = mcfi.get("mean_tpot_ms", 0)
            tpot_ratio = mcfi_tpot / fi_tpot if fi_tpot > 0 else 0
            speedup_html += f"""<tr>
                <td><b>{tier}</b></td>
                <td class="num {'good' if ratio > 1 else 'bad'}">{ratio:.2f}x</td>
                <td class="num">{fi_out:.1f} vs {mcfi_out:.1f}</td>
                <td class="num {'good' if ttft_ratio > 1 else 'bad'}">{ttft_ratio:.2f}x slower</td>
                <td class="num">{fi_ttft:.0f} vs {mcfi_ttft:.0f}</td>
                <td class="num {'good' if tpot_ratio > 1 else 'bad'}">{tpot_ratio:.2f}x slower</td>
                <td class="num">{fi_tpot:.2f} vs {mcfi_tpot:.2f}</td>
            </tr>\n"""

    # Build prefill vs decode time split for SALA configs
    prefill_decode_html = ""
    for config_key, sdata in data["sala"].items():
        if not sdata:
            continue
        tp = sdata["total_prefill_ms"]
        td = sdata["total_decode_ms"]
        total = tp + td
        if total == 0:
            continue
        prefill_decode_html += f"""<tr>
            <td><b>{config_key}</b></td>
            <td class="num">{total:.0f}</td>
            <td class="num">{tp:.0f} ({tp/total*100:.1f}%)</td>
            <td class="num">{td:.0f} ({td/total*100:.1f}%)</td>
            <td><div class="stacked-bar">
                <div class="stacked-prefill" style="width:{tp/total*100:.1f}%">{tp/total*100:.0f}%</div>
                <div class="stacked-decode" style="width:{td/total*100:.1f}%">{td/total*100:.0f}%</div>
            </div></td>
        </tr>\n"""

    # Top bottlenecks across all configs
    bottleneck_html = ""
    for config_key, sdata in data["sala"].items():
        if not sdata:
            continue
        cd = sdata["component_data"]
        grand = sdata["grand_total_ms"]
        # Sort all components by total time
        all_comps = []
        for ck, cv in cd.items():
            total = cv["prefill_ms"] + cv["decode_ms"]
            all_comps.append((ck, total, cv["prefill_ms"], cv["decode_ms"]))
        all_comps.sort(key=lambda x: -x[1])

        rows = ""
        for i, (ck, total, pf, dc) in enumerate(all_comps[:10]):
            pct = total / grand * 100 if grand > 0 else 0
            rows += f"""<tr>
                <td>{i+1}</td><td>{ck}</td>
                <td class="num">{total:.1f}</td>
                <td class="num">{pf:.1f}</td>
                <td class="num">{dc:.1f}</td>
                <td class="num">{pct:.1f}%</td>
                <td><div class="bar-container"><div class="bar bar-hot" style="width:{min(pct*2,100):.1f}%"></div></div></td>
            </tr>\n"""
        bottleneck_html += f"""
        <div class="bottleneck-section">
            <h4>{config_key} - Top 10 Bottlenecks (Total: {grand:.0f} ms)</h4>
            <table class="data-table">
                <thead><tr><th>#</th><th>Component</th><th>Total (ms)</th><th>Prefill</th><th>Decode</th><th>% of Grand</th><th></th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MiniCPM-SALA Profiling Dashboard</title>
<style>
:root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
    --lightning: #3fb950; --minicpm4: #f0883e; --hot: #f85149;
    --prefill: #58a6ff; --decode: #bc8cff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', monospace; font-size: 13px; padding: 20px; }}
h1 {{ font-size: 22px; margin-bottom: 5px; color: var(--accent); }}
h2 {{ font-size: 16px; margin: 25px 0 10px; padding: 8px 0; border-bottom: 1px solid var(--border); color: var(--text); }}
h3 {{ font-size: 14px; margin: 15px 0 8px; color: var(--text2); }}
h4 {{ font-size: 13px; margin: 12px 0 6px; color: var(--text2); }}
.subtitle {{ color: var(--text2); margin-bottom: 20px; font-size: 12px; }}
.tab-bar {{ display: flex; gap: 2px; margin-bottom: 15px; flex-wrap: wrap; }}
.tab {{ padding: 6px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; cursor: pointer; color: var(--text2); font-size: 12px; }}
.tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
.data-table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; font-size: 12px; }}
.data-table th {{ background: var(--surface); padding: 6px 8px; text-align: left; border: 1px solid var(--border); color: var(--text2); font-weight: 600; position: sticky; top: 0; }}
.data-table td {{ padding: 5px 8px; border: 1px solid var(--border); }}
.data-table tr:hover {{ background: rgba(88,166,255,0.08); }}
.data-table.compact td, .data-table.compact th {{ padding: 3px 6px; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; }}
.badge-lightning {{ background: rgba(63,185,80,0.15); color: var(--lightning); }}
.badge-minicpm4 {{ background: rgba(240,136,62,0.15); color: var(--minicpm4); }}
.bar-container {{ width: 120px; height: 14px; background: var(--surface); border-radius: 2px; overflow: hidden; }}
.bar {{ height: 100%; border-radius: 2px; }}
.bar-lightning {{ background: var(--lightning); }}
.bar-minicpm4 {{ background: var(--minicpm4); }}
.bar-hot {{ background: var(--hot); }}
.stacked-bar {{ display: flex; height: 20px; border-radius: 3px; overflow: hidden; width: 200px; font-size: 10px; line-height: 20px; text-align: center; }}
.stacked-prefill {{ background: var(--prefill); color: #fff; }}
.stacked-decode {{ background: var(--decode); color: #fff; }}
.summary-row {{ display: flex; gap: 15px; margin-bottom: 10px; flex-wrap: wrap; font-size: 12px; color: var(--text2); }}
.summary-row span {{ padding: 3px 8px; background: var(--surface); border-radius: 4px; }}
.good {{ color: var(--lightning); }}
.bad {{ color: var(--hot); }}
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
.card h4 {{ margin-top: 0; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-bottom: 15px; }}
.metric-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }}
.metric-card .label {{ font-size: 10px; color: var(--text2); text-transform: uppercase; }}
.metric-card .value {{ font-size: 20px; font-weight: 700; margin-top: 2px; }}
.metric-card .sub {{ font-size: 10px; color: var(--text2); }}
.layer-row {{ cursor: pointer; }}
.layer-row.minicpm4 {{ background: rgba(240,136,62,0.04); }}
.layer-row.lightning {{ background: rgba(63,185,80,0.04); }}
.detail-layer {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px; margin: 8px 0; }}
.scroll-table {{ max-height: 600px; overflow-y: auto; }}
.config-filter {{ margin-bottom: 10px; }}
.config-filter select {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); padding: 4px 8px; border-radius: 4px; }}
</style>
</head>
<body>

<h1>MiniCPM-SALA Profiling Dashboard</h1>
<p class="subtitle">Model: MiniCPM-SALA 9.48B (32 layers: 8 minicpm4 + 24 lightning-attn) | GPTQ-Int4 | Backends: flashinfer vs minicpm_flashinfer | Tiers: c1/c8/c64</p>

<div class="tab-bar" id="mainTabs">
    <div class="tab active" data-tab="overview">Overview</div>
    <div class="tab" data-tab="benchmarks">Benchmarks</div>
    <div class="tab" data-tab="bottlenecks">Bottlenecks</div>
    <div class="tab" data-tab="layers">Per-Layer</div>
    <div class="tab" data-tab="components">Components</div>
    <div class="tab" data-tab="detail">Layer Detail</div>
</div>

<!-- ═══════════ OVERVIEW TAB ═══════════ -->
<div class="tab-content active" id="tab-overview">
    <h2>Backend Comparison: flashinfer vs minicpm_flashinfer</h2>
    <table class="data-table">
        <thead><tr>
            <th>Tier</th><th>Output Tput Ratio (fi/mcfi)</th><th>Output tok/s</th>
            <th>TTFT Ratio</th><th>Mean TTFT (ms)</th>
            <th>TPOT Ratio</th><th>Mean TPOT (ms)</th>
        </tr></thead>
        <tbody>{speedup_html}</tbody>
    </table>

    <h2>Prefill vs Decode Time Split (SALA Profiler)</h2>
    <table class="data-table">
        <thead><tr><th>Config</th><th>Grand Total (ms)</th><th>Prefill</th><th>Decode</th><th>Split</th></tr></thead>
        <tbody>{prefill_decode_html}</tbody>
    </table>

    <h2>Architecture Reminder</h2>
    <div class="summary-row">
        <span>32 layers total</span>
        <span class="badge badge-minicpm4">8 minicpm4 layers (21.4% params): L0, L9, L16-17, L22, L29-31</span>
        <span class="badge badge-lightning">24 lightning-attn layers (72.2% params): all others</span>
        <span>MLP: 68% of params | Attention: 26% | Embeddings: 6%</span>
    </div>
    <div class="summary-row">
        <span>minicpm4: GQA 32 heads / 2 KV heads (16:1), sparse attn, no RoPE</span>
        <span>lightning: Full MHA 32/32 heads, linear attn (FLA), RoPE, z_proj gating</span>
    </div>
</div>

<!-- ═══════════ BENCHMARKS TAB ═══════════ -->
<div class="tab-content" id="tab-benchmarks">
    <h2>Full Benchmark Results</h2>
    <div class="scroll-table">
    <table class="data-table">
        <thead><tr>
            <th>Backend</th><th>Tier</th>
            <th>Total Tput</th><th>Input Tput</th><th>Output Tput</th><th>Peak Out</th>
            <th>TTFT Mean</th><th>TTFT Med</th><th>TTFT P99</th>
            <th>TPOT Mean</th><th>TPOT Med</th>
            <th>ITL Mean</th><th>ITL Med</th><th>ITL P99</th><th>ITL Max</th>
            <th>Duration</th><th>Conc</th><th>SM%</th><th>Peak Mem</th>
        </tr></thead>
        <tbody>{bench_rows}</tbody>
    </table>
    </div>
</div>

<!-- ═══════════ BOTTLENECKS TAB ═══════════ -->
<div class="tab-content" id="tab-bottlenecks">
    <h2>Top Bottlenecks by Component</h2>
    {bottleneck_html}
</div>

<!-- ═══════════ LAYERS TAB ═══════════ -->
<div class="tab-content" id="tab-layers">
    <h2>Per-Layer Timing (click row for detail)</h2>
    <div class="scroll-table">
    {layer_tables_html}
    </div>
</div>

<!-- ═══════════ COMPONENTS TAB ═══════════ -->
<div class="tab-content" id="tab-components">
    <h2>Component Breakdown by Layer Type</h2>
    {component_tables_html}
</div>

<!-- ═══════════ DETAIL TAB ═══════════ -->
<div class="tab-content" id="tab-detail">
    <h2>Per-Layer Component Detail</h2>
    <p style="color:var(--text2)">Click a layer row in the "Per-Layer" tab, or use the selector below:</p>
    <div class="config-filter">
        <select id="detailConfig">
            {"".join(f'<option value="{k}">{k}</option>' for k in data["sala"] if data["sala"][k])}
        </select>
        <select id="detailLayer">
            {"".join(f'<option value="{i}">L{i:02d} ({"minicpm4" if i in MINICPM4_LAYERS else "lightning"})</option>' for i in range(32))}
        </select>
        <button onclick="showDetail()">Show</button>
    </div>
    <div id="detailContainer">
        {detail_tables_html}
    </div>
</div>

<script>
// Tab switching
document.querySelectorAll('#mainTabs .tab').forEach(tab => {{
    tab.addEventListener('click', () => {{
        document.querySelectorAll('#mainTabs .tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    }});
}});

// Layer row click -> show detail
document.querySelectorAll('.layer-row').forEach(row => {{
    row.addEventListener('click', () => {{
        const ln = row.dataset.layer;
        const config = row.dataset.config;
        // Switch to detail tab
        document.querySelectorAll('#mainTabs .tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.querySelector('[data-tab="detail"]').classList.add('active');
        document.getElementById('tab-detail').classList.add('active');
        // Show matching detail
        document.querySelectorAll('.detail-layer').forEach(d => d.style.display = 'none');
        const target = document.querySelector(`.detail-layer[data-config="${{config}}"][data-layer="${{ln}}"]`);
        if (target) target.style.display = 'block';
        // Update selectors
        document.getElementById('detailConfig').value = config;
        document.getElementById('detailLayer').value = ln;
    }});
}});

function showDetail() {{
    const config = document.getElementById('detailConfig').value;
    const ln = document.getElementById('detailLayer').value;
    document.querySelectorAll('.detail-layer').forEach(d => d.style.display = 'none');
    const target = document.querySelector(`.detail-layer[data-config="${{config}}"][data-layer="${{ln}}"]`);
    if (target) target.style.display = 'block';
}}

// Embedded data for console/future use
window.BENCH_DATA = {json.dumps(bench_json, indent=2)};
window.SALA_DATA = {json.dumps(sala_json, indent=2)};
window.GPU_DATA = {json.dumps(gpu_json, indent=2)};
</script>

</body>
</html>"""

    return html


def main():
    print("Collecting profiling data...")
    data = collect_all_data()

    print("Generating HTML dashboard...")
    html = generate_html(data)

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"Dashboard written to: {OUTPUT_HTML}")
    print(f"  Benchmarks: {sum(1 for v in data['benchmarks'].values() if v)}/6")
    print(f"  SALA profiles: {sum(1 for v in data['sala'].values() if v)}/6")
    print(f"  GPU analyses: {sum(1 for v in data['gpu'].values() if v)}/6")


if __name__ == "__main__":
    main()
