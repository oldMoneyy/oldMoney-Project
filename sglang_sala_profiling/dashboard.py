import os
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = "results"
OUTPUT_HTML = "MiniCPM_SALA_Full_Report.html"

# ==============================================================================
# 1. PARSING FUNCTIONS
# ==============================================================================
def parse_bench_file(filepath):
    metrics = {}
    if not os.path.exists(filepath): return metrics
    with open(filepath, 'r') as f: text = f.read()
    patterns = {
        'Total Throughput': r"Total token throughput.*:\s+([\d.]+)",
        'TTFT': r"Mean TTFT \(ms\):\s+([\d.]+)",
        'TPOT': r"Mean TPOT \(ms\):\s+([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match: metrics[key] = float(match.group(1))
    return metrics

def parse_gpu_file(filepath):
    metrics = {}
    if not os.path.exists(filepath): return metrics
    with open(filepath, 'r') as f: text = f.read()
    sm_match = re.search(r"Avg SM%:\s+([\d.]+)%", text)
    mem_peak_match = re.search(r"Peak:\s+(\d+)\s+MiB", text)
    if sm_match: metrics['Avg SM'] = float(sm_match.group(1))
    if mem_peak_match: metrics['Peak VRAM'] = float(mem_peak_match.group(1))
    return metrics

def parse_sala_csv(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    df = pd.read_csv(filepath)
    def parse_key(key):
        parts = str(key).split('/')
        phase = parts[-1]
        if parts[0].startswith('L') and len(parts[0]) <= 3:
            return int(parts[0].replace('L', '')), parts[1], "/".join(parts[2:-1]), phase
        else:
            return -1, parts[0], "/".join(parts[1:-1]), phase
    parsed = df['key'].apply(parse_key).apply(pd.Series)
    parsed.columns = ['Layer', 'Layer_Type', 'Component', 'Phase']
    return pd.concat([df, parsed], axis=1)

def collect_data():
    bench_data, sala_data = [], []
    for backend in ['flashinfer', 'minicpm_flashinfer']:
        for tier in ['c1', 'c8', 'c64']:
            tier_dir = os.path.join(RESULTS_DIR, backend, tier)
            if not os.path.exists(tier_dir): continue
            
            bench = parse_bench_file(os.path.join(tier_dir, 'bench_baseline.txt'))
            gpu = parse_gpu_file(os.path.join(tier_dir, 'gpu_analysis.txt'))
            if bench:
                row = {'Backend': backend, 'Tier': tier}
                row.update(bench); row.update(gpu)
                bench_data.append(row)
                
            sala_df = parse_sala_csv(os.path.join(tier_dir, 'sala_timings.csv'))
            if not sala_df.empty:
                sala_df['Backend'] = backend; sala_df['Tier'] = tier
                sala_data.append(sala_df)
    return pd.DataFrame(bench_data), pd.concat(sala_data, ignore_index=True) if sala_data else pd.DataFrame()

# ==============================================================================
# 2. PLOTLY GENERATION (STYLED TO MATCH YOUR CSS)
# ==============================================================================
def create_throughput_chart(bench_df):
    if bench_df.empty: return ""
    fig = go.Figure()
    colors = {'flashinfer': '#0E7490', 'minicpm_flashinfer': '#3B82F6'} # Teal and Blue
    for backend in bench_df['Backend'].unique():
        b_df = bench_df[bench_df['Backend'] == backend].sort_values('Tier')
        fig.add_trace(go.Bar(x=b_df['Tier'], y=b_df['Total Throughput'], name=backend, marker_color=colors[backend]))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#5E5D5A'),
        title="Total Throughput (tok/s) by Concurrency",
        barmode='group', height=350, margin=dict(t=40, b=20, l=20, r=20),
        yaxis=dict(gridcolor='rgba(20,20,19,0.06)')
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_layer_chart(sala_df):
    if sala_df.empty: return ""
    c8_layers = sala_df[(sala_df['Tier'] == 'c8') & (sala_df['Layer'] >= 0)]
    if c8_layers.empty: return ""
    
    layer_times = c8_layers.groupby(['Backend', 'Layer', 'Phase', 'Layer_Type'])['total_ms'].sum().reset_index()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Prefill Time per Layer (ms)", "Decode Time per Layer (ms)"))
    
    # Matching your CSS colors: minicpm4 = Terra, lightning-attn = Blue
    layer_colors = {'minicpm4': '#9F1239', 'lightning-attn': '#3B82F6'} 
    
    for backend in layer_times['Backend'].unique():
        b_df = layer_times[layer_times['Backend'] == backend]
        
        prefill = b_df[b_df['Phase'] == 'prefill']
        marker_colors = [layer_colors.get(t, 'gray') for t in prefill['Layer_Type']]
        fig.add_trace(go.Bar(x=prefill['Layer'], y=prefill['total_ms'], name=f"{backend} Prefill", 
                             marker_color=marker_colors, hovertext=prefill['Layer_Type'], showlegend=False), row=1, col=1)
        
        decode = b_df[b_df['Phase'] == 'decode']
        marker_colors = [layer_colors.get(t, 'gray') for t in decode['Layer_Type']]
        fig.add_trace(go.Bar(x=decode['Layer'], y=decode['total_ms'], name=f"{backend} Decode", 
                             marker_color=marker_colors, hovertext=decode['Layer_Type'], showlegend=False), row=2, col=1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#5E5D5A'),
        barmode='group', height=500, margin=dict(t=40, b=40, l=20, r=20),
        yaxis=dict(gridcolor='rgba(20,20,19,0.06)'), yaxis2=dict(gridcolor='rgba(20,20,19,0.06)'),
        xaxis2_title="Layer Index (0-31)"
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def create_pie_charts(sala_df):
    if sala_df.empty: return ""
    c8_comps = sala_df[(sala_df['Tier'] == 'c8') & (sala_df['Layer'] >= 0)]
    if c8_comps.empty: return ""
    
    comp_times = c8_comps.groupby(['Backend', 'Phase', 'Component'])['total_ms'].sum().reset_index()
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=("Flashinfer Prefill Breakdown", "Flashinfer Decode Breakdown"))
    
    prefill = comp_times[(comp_times['Backend'] == 'flashinfer') & (comp_times['Phase'] == 'prefill')]
    decode = comp_times[(comp_times['Backend'] == 'flashinfer') & (comp_times['Phase'] == 'decode')]
    
    if not prefill.empty:
        fig.add_trace(go.Pie(labels=prefill['Component'], values=prefill['total_ms'], textinfo='percent'), row=1, col=1)
    if not decode.empty:
        fig.add_trace(go.Pie(labels=decode['Component'], values=decode['total_ms'], textinfo='percent'), row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#5E5D5A'),
        height=350, margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

# ==============================================================================
# 3. HTML ASSEMBLY
# ==============================================================================
BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MiniCPM-SALA — Quantization & Profiling Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&family=Playfair+Display:wght@400;500;600;700&display=swap');
:root{--bg:#F3F1EC;--bg2:#FFFFFF;--bg3:#FAF9F6;--txt:#141413;--txt2:#5E5D5A;--txt3:#8A8884;--txtf:#B0AEA9;--bdr:rgba(20,20,19,.12);--bdr2:rgba(20,20,19,.06);--amber:#D97706;--amberbg:rgba(217,119,6,.06);--amberbr:rgba(217,119,6,.2);--moss:#4D7C0F;--mossbg:rgba(77,124,15,.06);--mossbr:rgba(77,124,15,.2);--blue:#3B82F6;--bluebg:rgba(59,130,246,.06);--bluebr:rgba(59,130,246,.18);--terra:#9F1239;--terrabg:rgba(159,18,57,.05);--terrabr:rgba(159,18,57,.18);--pink:#DB2777;--pinkbg:rgba(219,39,119,.05);--pinkbr:rgba(219,39,119,.18);--violet:#6D28D9;--violetbg:rgba(109,40,217,.04);--violetbr:rgba(109,40,217,.18);--teal:#0E7490;--tealbg:rgba(14,116,144,.05);--tealbr:rgba(14,116,144,.18);--serif:'Playfair Display',Georgia,serif;--sans:'Inter',system-ui,sans-serif;--mono:'IBM Plex Mono',monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:rgba(20,20,19,.2);border-radius:4px}
.pg{max-width:1400px;margin:0 auto;padding:56px 40px 100px}
.hdr{text-align:center;padding:48px 0 40px;border-bottom:1px solid var(--bdr);margin-bottom:48px}
.hdr h1{font-family:var(--serif);font-size:42px;font-weight:700;color:var(--txt);letter-spacing:-.03em;margin-bottom:6px}
.hdr .sub{font-family:var(--sans);font-size:15px;color:var(--txt2);line-height:1.6;max-width:720px;margin:0 auto}
.hdr .specs{font-family:var(--mono);font-size:13px;color:var(--txt3);margin-top:16px;display:flex;justify-content:center;gap:22px;flex-wrap:wrap}
.SEC{border:1px solid var(--bdr);border-radius:12px;padding:28px 32px;margin:0 0 32px;background:var(--bg2);box-shadow:0 1px 3px rgba(0,0,0,.03)}
.SEC-t{font-family:var(--serif);font-size:22px;font-weight:600;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid var(--bdr2)}
.SEC-t .tag{font-family:var(--mono);font-size:12px;font-weight:500;padding:3px 10px;border-radius:5px;margin-left:10px;vertical-align:middle;display:inline-block;position:relative;top:-2px}
.SEC-t .tag.fp4{background:var(--pinkbg);color:var(--pink);border:1px solid var(--pinkbr)}
.SEC-t .tag.int4{background:var(--tealbg);color:var(--teal);border:1px solid var(--tealbr)}
.SEC-t .tag.calib{background:var(--amberbg);color:var(--amber);border:1px solid var(--amberbr)}
.SEC-t .tag.run{background:var(--mossbg);color:var(--moss);border:1px solid var(--mossbr)}
.SEC-t .tag.cmp{background:var(--violetbg);color:var(--violet);border:1px solid var(--violetbr)}
.SEC-t .tag.prof{background:var(--bluebg);color:var(--blue);border:1px solid var(--bluebr)}
.body-text{font-family:var(--sans);font-size:14.5px;color:var(--txt2);line-height:1.8;margin:10px 0}
.tbl-wrap{overflow-x:auto;margin:16px 0}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12.5px}
thead th{background:var(--bg3);font-weight:600;color:var(--txt);text-align:left;padding:10px 16px;border-bottom:2px solid var(--bdr);white-space:nowrap}
tbody td{padding:10px 16px;border-bottom:1px solid var(--bdr2);color:var(--txt2);vertical-align:top}
tbody tr:hover{background:var(--bg3)}
td.em{font-weight:600;color:var(--txt)}
.sub-sec{margin:20px 0 16px;font-family:var(--serif);font-size:16px;font-weight:600;color:var(--txt)}
.T{display:inline-flex;align-items:center;padding:3px 10px;border-radius:5px;font-family:var(--mono);font-size:12px;font-weight:500;white-space:nowrap;border:1px solid;transition:box-shadow .15s ease}
.T:hover{box-shadow:0 1px 4px rgba(0,0,0,.06)}
.Th{background:var(--bluebg);border-color:var(--bluebr);color:var(--blue)}
.Tq{background:var(--pinkbg);border-color:var(--pinkbr);color:var(--pink)}
.Tk{background:var(--amberbg);border-color:var(--amberbr);color:var(--amber)}
.Tv{background:var(--mossbg);border-color:var(--mossbr);color:var(--moss)}
.To{background:var(--violetbg);border-color:var(--violetbr);color:var(--violet)}
.Ts{background:var(--tealbg);border-color:var(--tealbr);color:var(--teal)}
.Tg{background:var(--terrabg);border-color:var(--terrabr);color:var(--terra)}
.codeblock{background:var(--bg3);border:1px solid var(--bdr);border-radius:8px;padding:16px 20px;margin:14px 0;font-family:var(--mono);font-size:12.5px;line-height:1.7;overflow-x:auto;color:var(--txt2);white-space:pre-wrap;word-break:break-all}
.codeblock .cmd{color:var(--moss);font-weight:500}
.codeblock .flag{color:var(--amber)}
.codeblock .val{color:var(--blue)}
.codeblock .cmt{color:var(--txt3);font-style:italic}
.note{font-family:var(--mono);font-size:12px;color:var(--amber);padding:10px 18px;border:1px dashed var(--amberbr);border-radius:8px;background:var(--amberbg);margin:14px 0;line-height:1.7}
.note.teal{color:var(--teal);border-color:var(--tealbr);background:var(--tealbg)}
.note.pink{color:var(--pink);border-color:var(--pinkbr);background:var(--pinkbg)}
.note.moss{color:var(--moss);border-color:var(--mossbr);background:var(--mossbg)}
.ann{font-family:var(--mono);font-size:11px;color:var(--txt3);text-align:center;margin:6px 0;line-height:1.6}
.dual{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:16px 0}
@media(max-width:900px){.dual{grid-template-columns:1fr}}
.stat-row{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}
.stat{flex:1;min-width:140px;border:1px solid var(--bdr);border-radius:8px;padding:14px 18px;background:var(--bg3);text-align:center}
.stat .v{font-family:var(--serif);font-size:26px;font-weight:700;color:var(--txt);line-height:1.2}
.stat .l{font-family:var(--mono);font-size:11px;color:var(--txt3);margin-top:4px}
.dist-bar{display:flex;height:28px;border-radius:6px;overflow:hidden;margin:10px 0;border:1px solid var(--bdr)}
.dist-bar .seg{display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;font-weight:600;color:#fff;min-width:20px;transition:flex .3s ease}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;justify-content:center}
.legend-item{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--txt2)}
.legend-dot{width:10px;height:10px;border-radius:3px}
.sep{height:1px;background:var(--bdr2);margin:18px 0}
</style>
</head>
<body>
<div class="pg">

<!-- ========== HEADER ========== -->
<div class="hdr">
  <h1>MiniCPM-SALA Optimization Report</h1>
  <div class="sub">Part 1: Post-Training Quantization Strategy<br>Part 2: Hardware & Attention Backend Profiling</div>
  <div class="specs">
    <span>9.48B params</span>
    <span>32 layers (8 sparse + 24 linear)</span>
    <span>hidden 4096</span>
    <span>SALA hybrid architecture</span>
  </div>
</div>

<!-- ========== PART 1: QUANTIZATION STRATEGY (YOUR ORIGINAL CONTENT) ========== -->
<div class="SEC">
  <div class="SEC-t">Script 1 — AWQ NVFP4 <span class="tag fp4">E2M1 + FP8 Block Scales</span></div>
  <div class="ann">AWQ_L_4_Mini_16_smoothed.py — Targeting NVIDIA Blackwell GPUs via SGLang's ModelOptFp4Config</div>
  <div class="sub-sec">Architecture-Aware Mixed Precision</div>
  <div class="body-text">
    The critical design decision is driven by SGLang's fused kernel constraints: SGLang fuses q+k+v into one QKV kernel, requiring all projections to match precision.
    Since MiniCPM4's k/v projections are only 256×4096 (too small for FP4 to be accurate), the <em>entire</em> attention block in those 8 layers stays in BF16.
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Layer Type</th><th>Attention</th><th>MLP</th><th>Rationale</th></tr></thead>
      <tbody>
        <tr>
          <td class="em"><span class="T Tg" style="font-size:11px">minicpm4</span> × 8</td>
          <td><span class="T Th" style="font-size:11px">BF16 (excluded)</span></td>
          <td><span class="T Tq" style="font-size:11px">FP4</span></td>
          <td>k/v only 256-dim — too small for FP4; fused QKV kernel requires uniform precision</td>
        </tr>
        <tr>
          <td class="em"><span class="T Tq" style="font-size:11px">lightning-attn</span> × 24</td>
          <td><span class="T Tq" style="font-size:11px">FP4 + smoothing</span></td>
          <td><span class="T Tq" style="font-size:11px">FP4 + smoothing</span></td>
          <td>Full MHA with 4096-dim k/v — all projections (q,k,v,z,o) large enough for FP4</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<div class="SEC">
  <div class="SEC-t">Script 2 — GPTQ W4A16 <span class="tag int4">INT4 Symmetric + Marlin</span></div>
  <div class="ann">GPTQ_int4_flashinfer_dense_smoothing_gpu.py — Targeting FlashInfer / gptq_marlin kernels</div>
  <div class="sub-sec">Architecture-Aware Quantization</div>
  <div class="body-text">
    Unlike AWQ NVFP4 which excludes minicpm4 attention, GPTQ quantizes <strong>all linears in all 32 layers</strong> uniformly to INT4. This works because gptq_marlin doesn't require fused QKV precision matching.
  </div>
  <div class="tbl-wrap">
    <table>
      <thead><tr><th>Layer Type</th><th>Attention</th><th>MLP</th><th>Rationale</th></tr></thead>
      <tbody>
        <tr>
          <td class="em"><span class="T Tg" style="font-size:11px">minicpm4</span> × 8</td>
          <td><span class="T Ts" style="font-size:11px">INT4</span></td>
          <td><span class="T Ts" style="font-size:11px">INT4</span></td>
          <td>No fused QKV constraint in Marlin; full Hessian error propagation compensates for small dims</td>
        </tr>
        <tr>
          <td class="em"><span class="T Tq" style="font-size:11px">lightning-attn</span> × 24</td>
          <td><span class="T Ts" style="font-size:11px">INT4 + smoothing</span></td>
          <td><span class="T Ts" style="font-size:11px">INT4 + smoothing</span></td>
          <td>All projections quantized with 4-group smoothing including up→down output-dim</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ======================================================================= -->
<!-- INSERT POINT: PROFILING DATA                                            -->
<!-- ======================================================================= -->
<!-- PROFILING_SECTION_PLACEHOLDER -->

<!-- ========== FOOTER ========== -->
<div style="text-align:center;padding:36px;font-size:13px;color:var(--txt3)">
  MiniCPM-SALA 9.48B · Quantization & Profiling Report<br>
  Generated via Automated SGLang Benchmarks & CUDA Events<br>
</div>

</div>
</body>
</html>"""

def generate_report():
    print("Parsing SGLang Profiler data...")
    bench_df, sala_df = collect_data()
    
    # Generate Plotly HTML strings
    chart_throughput = create_throughput_chart(bench_df)
    chart_layers = create_layer_chart(sala_df)
    chart_pie = create_pie_charts(sala_df)
    
    # Build the HTML for the Tables based on DataFrames
    table_rows = ""
    if not bench_df.empty:
        for _, row in bench_df.iterrows():
            table_rows += f"""
            <tr>
                <td class="em">{row['Backend']}</td>
                <td>{row['Tier']}</td>
                <td>{row.get('Total Throughput', 0):.2f}</td>
                <td>{row.get('TTFT', 0):.2f} ms</td>
                <td>{row.get('TPOT', 0):.2f} ms</td>
                <td>{row.get('Peak VRAM', 0):.0f} MB</td>
            </tr>"""

    # Construct the Profiling Section using your CSS classes
    profiling_html = f"""
    <div class="hdr" style="padding-bottom: 20px; border-bottom: none; margin-bottom: 20px; margin-top: 40px;">
        <h1>Part 2: Hardware & Backend Profiling</h1>
        <div class="sub">Performance breakdown of <b>Flashinfer</b> vs <b>Minicpm_Flashinfer</b> across concurrency tiers (c1, c8, c64).</div>
    </div>

    <!-- SGLang Serving Performance -->
    <div class="SEC">
      <div class="SEC-t">End-to-End Serving Performance <span class="tag prof">SGLang Benchmark</span></div>
      
      <div class="dual">
          <div>
              <div class="sub-sec" style="margin-top:0;">Throughput Scaling</div>
              {chart_throughput}
          </div>
          <div>
              <div class="sub-sec" style="margin-top:0;">Metrics Table</div>
              <div class="tbl-wrap">
                <table>
                  <thead><tr><th>Backend</th><th>Tier</th><th>Tok/s</th><th>TTFT</th><th>TPOT</th><th>Peak VRAM</th></tr></thead>
                  <tbody>
                    {table_rows}
                  </tbody>
                </table>
              </div>
              <div class="note teal">
                <strong>Analysis:</strong> Look for how well the backends scale as concurrency increases from 1 to 64. A drop in TPOT or flat throughput indicates a memory-bandwidth bottleneck.
              </div>
          </div>
      </div>
    </div>

    <!-- Architecture Layer Breakdown -->
    <div class="SEC">
      <div class="SEC-t">Architecture Bottleneck: Sparse vs Linear <span class="tag run">CUDA Events</span></div>
      <div class="ann">Execution time per layer for Concurrency=8 (Prefill vs Decode)</div>
      
      <div class="body-text">
        MiniCPM-SALA mixes 8 Sparse Attention layers (<span style="color:var(--terra);font-weight:600;">minicpm4</span>) with 24 Linear Attention layers (<span style="color:var(--blue);font-weight:600;">lightning-attn</span>). This chart reveals exactly which layer type creates the bottleneck during the Prefill and Decode phases.
      </div>
      
      <div style="margin-top: 20px; margin-bottom: 20px;">
          <div class="legend" style="justify-content: flex-start; margin-bottom: 10px;">
            <div class="legend-item"><div class="legend-dot" style="background:var(--terra)"></div>Minicpm4 (Sparse Layers: 0, 9, 16, 17, 22, 29, 30, 31)</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--blue)"></div>Lightning-attn (Linear Layers)</div>
          </div>
          {chart_layers}
      </div>
    </div>

    <!-- Component Breakdown -->
    <div class="SEC">
      <div class="SEC-t">Time Distribution by Component <span class="tag cmp">Profile Breakdown</span></div>
      <div class="body-text">
        Where does the GPU spend its time? If MLP dominates, the model is compute-bound. If <code>attn_kernel</code> dominates, the model is memory-bound (or the kernel implementation is inefficient).
      </div>
      <div style="margin-top: 20px;">
          {chart_pie}
      </div>
    </div>
    """

    # Inject into the base HTML
    final_html = BASE_HTML.replace("<!-- PROFILING_SECTION_PLACEHOLDER -->", profiling_html)
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"✅ Integrated report generated successfully: {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_report()
