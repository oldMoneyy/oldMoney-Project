import os
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

RESULTS_DIR = "results"
OUTPUT_HTML = "sala_profiling_dashboard.html"

def parse_bench_file(filepath):
    metrics = {}
    if not os.path.exists(filepath):
        return metrics
    with open(filepath, 'r') as f:
        text = f.read()
        
    patterns = {
        'Total Throughput (tok/s)': r"Total token throughput.*:\s+([\d.]+)",
        'Out Throughput (tok/s)': r"Output token throughput.*:\s+([\d.]+)",
        'TTFT Mean (ms)': r"Mean TTFT \(ms\):\s+([\d.]+)",
        'TPOT Mean (ms)': r"Mean TPOT \(ms\):\s+([\d.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics

def parse_gpu_file(filepath):
    metrics = {}
    if not os.path.exists(filepath):
        return metrics
    with open(filepath, 'r') as f:
        text = f.read()
        
    sm_match = re.search(r"Avg SM%:\s+([\d.]+)%", text)
    mem_peak_match = re.search(r"Peak:\s+(\d+)\s+MiB", text)
    
    if sm_match: metrics['Avg SM (%)'] = float(sm_match.group(1))
    if mem_peak_match: metrics['Peak VRAM (MB)'] = float(mem_peak_match.group(1))
    return metrics

def parse_sala_csv(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    
    # Parse key: L00/minicpm4/attn/decode OR non-layer/embed_tokens/prefill
    def parse_key(key):
        parts = str(key).split('/')
        phase = parts[-1]
        if parts[0].startswith('L') and len(parts[0]) <= 3:
            layer_idx = int(parts[0].replace('L', ''))
            layer_type = parts[1]
            component = "/".join(parts[2:-1])
            return layer_idx, layer_type, component, phase
        else:
            layer_type = parts[0]
            component = "/".join(parts[1:-1])
            return -1, layer_type, component, phase

    parsed = df['key'].apply(parse_key).apply(pd.Series)
    parsed.columns = ['Layer', 'Layer_Type', 'Component', 'Phase']
    df = pd.concat([df, parsed], axis=1)
    return df

def collect_data():
    bench_data = []
    sala_data = []
    
    for backend in ['flashinfer', 'minicpm_flashinfer']:
        for tier in ['c1', 'c8', 'c64']:
            tier_dir = os.path.join(RESULTS_DIR, backend, tier)
            if not os.path.exists(tier_dir):
                continue
                
            # 1. Parse Baseline
            bench = parse_bench_file(os.path.join(tier_dir, 'bench_baseline.txt'))
            gpu = parse_gpu_file(os.path.join(tier_dir, 'gpu_analysis.txt'))
            
            if bench:
                row = {'Backend': backend, 'Tier': tier}
                row.update(bench)
                row.update(gpu)
                bench_data.append(row)
                
            # 2. Parse SALA Profiler
            sala_df = parse_sala_csv(os.path.join(tier_dir, 'sala_timings.csv'))
            if not sala_df.empty:
                sala_df['Backend'] = backend
                sala_df['Tier'] = tier
                sala_data.append(sala_df)

    return pd.DataFrame(bench_data), pd.concat(sala_data, ignore_index=True) if sala_data else pd.DataFrame()

def create_dashboard(bench_df, sala_df):
    html_sections = []
    
    html_sections.append("""
    <html><head><title>MiniCPM-SALA Profiling Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
        h1, h2 { color: #58a6ff; text-align: center; border-bottom: 1px solid #30363d; padding-bottom: 10px;}
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head><body>
    <h1>🚀 MiniCPM-SALA Profiling Dashboard</h1>
    """)

    # ---------------------------------------------------------
    # CHART 1: Benchmark Metrics (Throughput & Latency)
    # ---------------------------------------------------------
    if not bench_df.empty:
        fig1 = make_subplots(rows=1, cols=3, subplot_titles=("Total Throughput (tok/s)", "Mean TTFT (ms)", "Mean TPOT (ms)"))
        colors = {'flashinfer': '#ff7b72', 'minicpm_flashinfer': '#3fb950'}
        
        for backend in bench_df['Backend'].unique():
            b_df = bench_df[bench_df['Backend'] == backend].sort_values('Tier')
            
            fig1.add_trace(go.Bar(x=b_df['Tier'], y=b_df['Total Throughput (tok/s)'], name=backend, marker_color=colors[backend], showlegend=True), row=1, col=1)
            fig1.add_trace(go.Bar(x=b_df['Tier'], y=b_df['TTFT Mean (ms)'], name=backend, marker_color=colors[backend], showlegend=False), row=1, col=2)
            fig1.add_trace(go.Bar(x=b_df['Tier'], y=b_df['TPOT Mean (ms)'], name=backend, marker_color=colors[backend], showlegend=False), row=1, col=3)

        fig1.update_layout(template="plotly_dark", barmode='group', height=400, margin=dict(t=40, b=20, l=20, r=20))
        html_sections.append(f"<div class='card'><h2>1. End-to-End Performance</h2>{fig1.to_html(full_html=False, include_plotlyjs=False)}</div>")

    # ---------------------------------------------------------
    # CHART 2: The Layer Heatmap (c8 Tier)
    # ---------------------------------------------------------
    if not sala_df.empty:
        # Filter for layers only (0 to 31) and specifically c8 tier
        c8_layers = sala_df[(sala_df['Tier'] == 'c8') & (sala_df['Layer'] >= 0)]
        
        if not c8_layers.empty:
            # Group by Layer, Backend, Phase, Layer_Type
            layer_times = c8_layers.groupby(['Backend', 'Layer', 'Phase', 'Layer_Type'])['total_ms'].sum().reset_index()
            
            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                 subplot_titles=("Prefill Time per Layer (c8)", "Decode Time per Layer (c8)"))
            
            # Colors mapping based on the sparse vs linear architectures
            layer_colors = {'minicpm4': '#d73a4a', 'lightning-attn': '#0366d6'} 
            
            for backend in layer_times['Backend'].unique():
                b_df = layer_times[layer_times['Backend'] == backend]
                
                # Prefill
                prefill = b_df[b_df['Phase'] == 'prefill']
                marker_colors = [layer_colors.get(t, 'gray') for t in prefill['Layer_Type']]
                fig2.add_trace(go.Bar(x=prefill['Layer'], y=prefill['total_ms'], name=f"{backend} Prefill", 
                                      marker_color=marker_colors, hovertext=prefill['Layer_Type']), row=1, col=1)
                
                # Decode
                decode = b_df[b_df['Phase'] == 'decode']
                marker_colors = [layer_colors.get(t, 'gray') for t in decode['Layer_Type']]
                fig2.add_trace(go.Bar(x=decode['Layer'], y=decode['total_ms'], name=f"{backend} Decode", 
                                      marker_color=marker_colors, hovertext=decode['Layer_Type']), row=2, col=1)

            fig2.update_layout(template="plotly_dark", barmode='group', height=600,
                               xaxis2_title="Layer Index (0-31)", yaxis_title="Total Time (ms)", yaxis2_title="Total Time (ms)")
            
            # Add custom legend for layer types
            fig2.add_annotation(text="<b style='color:#d73a4a'>RED: Minicpm4 (Sparse)</b> | <b style='color:#0366d6'>BLUE: Lightning (Linear)</b>", 
                                xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False, font=dict(size=14))

            html_sections.append(f"<div class='card'><h2>2. Layer Execution Profile (Concurrency=8)</h2>{fig2.to_html(full_html=False, include_plotlyjs=False)}<p style='text-align:center'><i>Notice how Layers 0, 9, 16, 17, 22, 29, 30, 31 (Sparse Attention) take significantly different time compared to the Linear Attention layers.</i></p></div>")

    # ---------------------------------------------------------
    # CHART 3: Component Breakdown
    # ---------------------------------------------------------
    if not sala_df.empty:
        c8_comps = sala_df[(sala_df['Tier'] == 'c8') & (sala_df['Layer'] >= 0)]
        if not c8_comps.empty:
            comp_times = c8_comps.groupby(['Backend', 'Phase', 'Component'])['total_ms'].sum().reset_index()
            
            fig3 = make_subplots(rows=2, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]],
                                 subplot_titles=("Flashinfer Prefill", "Flashinfer Decode", "Minicpm_Flashinfer Prefill", "Minicpm_Flashinfer Decode"))
            
            def add_pie(backend, phase, row, col):
                data = comp_times[(comp_times['Backend'] == backend) & (comp_times['Phase'] == phase)]
                if not data.empty:
                    fig3.add_trace(go.Pie(labels=data['Component'], values=data['total_ms'], textinfo='label+percent'), row=row, col=col)

            add_pie('flashinfer', 'prefill', 1, 1)
            add_pie('flashinfer', 'decode', 1, 2)
            add_pie('minicpm_flashinfer', 'prefill', 2, 1)
            add_pie('minicpm_flashinfer', 'decode', 2, 2)
            
            fig3.update_layout(template="plotly_dark", height=800)
            html_sections.append(f"<div class='card'><h2>3. Time Distribution by Component (Concurrency=8)</h2>{fig3.to_html(full_html=False, include_plotlyjs=False)}</div>")

    html_sections.append("</body></html>")
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_sections))
    print(f"✅ Dashboard generated successfully: {OUTPUT_HTML}")

if __name__ == "__main__":
    print("Collecting data from results/...")
    bench_df, sala_df = collect_data()
    print(f"Found {len(bench_df)} benchmark records and {len(sala_df)} profiler records.")
    create_dashboard(bench_df, sala_df)