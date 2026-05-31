import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Simulasi Antrian Server",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown(
    """
<style>
.metric-card {
    background: var(--background-color);
    border: 1px solid rgba(100,181,246,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-label { font-size: 10px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 700; color: #64b5f6; line-height: 1; }
.metric-sub { font-size: 11px; color: #90a4ae; margin-top: 4px; }
.metric-good .metric-value { color: #4caf50; }
.metric-warn .metric-value { color: #ff9800; }
.metric-bad .metric-value { color: #f44336; }
.rho-card {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-family: monospace;
    font-size: 13px;
    border-left: 3px solid #64b5f6;
    background: var(--secondary-background-color);
}
.littles-card {
    border: 1px solid rgba(100,181,246,0.2);
    border-left: 4px solid #64b5f6;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    background: var(--secondary-background-color);
}
.littles-card .eq { font-size: 20px; font-weight: 700; color: #64b5f6; font-family: monospace; }
.alert-stable { background: rgba(46,125,50,0.15); border: 1px solid #2e7d32; border-radius: 8px; padding: 10px 16px; color: #a5d6a7; font-size: 13px; }
.alert-critical { background: rgba(198,40,40,0.15); border: 1px solid #c62828; border-radius: 8px; padding: 10px 16px; color: #ef9a9a; font-size: 13px; }
.alert-warn { background: rgba(230,81,0,0.15); border: 1px solid #e65100; border-radius: 8px; padding: 10px 16px; color: #ffcc80; font-size: 13px; }
.section-header { font-size: 12px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(100,181,246,0.15); }
</style>
""",
    unsafe_allow_html=True,
)


# ── Simulation logic ──────────────────────────────────────────────
def select_worker(algorithm, request_id, workers_queue, num_workers):
    if algorithm == "round_robin":
        return request_id % num_workers
    elif algorithm == "random":
        return np.random.randint(0, num_workers)
    elif algorithm == "least_connection":
        return int(np.argmin(workers_queue))
    return request_id % num_workers


def run_simulation(lambda_rate, num_workers, algorithm, duration, mu):
    results = []
    workers_queue = [0] * num_workers
    workers_busy_until = [0.0] * num_workers
    current_time = 0.0
    request_id = 0
    base_time = datetime.now()

    while current_time < duration:
        inter_arrival = np.random.exponential(1.0 / lambda_rate)
        current_time += inter_arrival
        if current_time > duration:
            break
        worker_id = select_worker(algorithm, request_id, workers_queue, num_workers)
        wait_time = max(0.0, workers_busy_until[worker_id] - current_time)
        service_time = np.random.exponential(1.0 / mu)
        start_service = current_time + wait_time
        finish_time = start_service + service_time
        workers_busy_until[worker_id] = finish_time
        workers_queue[worker_id] += 1
        results.append(
            {
                "request_id": request_id,
                "worker_id": worker_id,
                "arrive_time": round(current_time, 4),
                "wait_time": round(wait_time, 4),
                "service_time": round(service_time, 4),
                "total_time": round(wait_time + service_time, 4),
                "timestamp": (base_time + timedelta(seconds=current_time)).isoformat(),
            }
        )
        request_id += 1

    for r in results:
        workers_queue[r["worker_id"]] = max(0, workers_queue[r["worker_id"]] - 1)
    return results


def run_all_algorithms(lambda_rate, num_workers, duration, mu):
    return {
        algo: run_simulation(lambda_rate, num_workers, algo, duration, mu)
        for algo in ["round_robin", "random", "least_connection"]
    }


def plotly_dark(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1422",
        font_color="#b0bec5",
        title=dict(text=title, font=dict(color="#90caf9", size=13)),
        xaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        yaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="#0d1422", bordercolor="#1a2a3a"),
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")
    st.markdown(
        '<div class="section-header">Parameter Stokastik</div>', unsafe_allow_html=True
    )

    lambda_rate = st.slider("λ — Arrival Rate (req/s)", 0.5, 5.0, 2.0, 0.5)
    mu = st.slider("μ — Service Rate (req/s)", 0.5, 3.0, 1.0, 0.5)
    num_workers = st.slider("Jumlah Worker (c)", 1, 5, 3)

    rho = lambda_rate / (num_workers * mu)
    rho_color = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_status = (
        "✅ Stabil"
        if rho < 0.7
        else "⚠️ Mendekati kritis" if rho < 1 else "🔴 Tidak stabil!"
    )
    st.markdown(
        f"""
    <div class="rho-card">
        ρ = λ/(c·μ) = {lambda_rate:.1f}/({num_workers}×{mu:.1f}) = 
        <span style="color:{rho_color};font-weight:700;font-size:16px">{rho:.2f}</span><br>
        <span style="color:{rho_color};font-size:12px">{rho_status}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-header">Konfigurasi Simulasi</div>', unsafe_allow_html=True
    )
    algorithm = st.selectbox(
        "Algoritma Load Balancer",
        ["round_robin", "random", "least_connection"],
        format_func=lambda x: {
            "round_robin": "🔄 Round Robin",
            "random": "🎲 Random",
            "least_connection": "📉 Least Connection",
        }[x],
    )
    duration = st.slider("Durasi Simulasi (detik)", 10, 60, 30, 10)

    st.markdown("---")
    run = st.button("▶ Jalankan Simulasi")

    st.markdown(
        '<div class="section-header">Tentang Model</div>', unsafe_allow_html=True
    )
    st.markdown(
        """
    <div style="font-size:11px;color:#546e7a;line-height:1.7">
    <b style="color:#78909c">M/M/c Queue:</b><br>
    • Arrival: Distribusi Poisson (λ)<br>
    • Service: Distribusi Eksponensial (μ)<br>
    • c: Jumlah worker paralel<br>
    • ρ &lt; 1 → sistem stabil
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── MAIN ──────────────────────────────────────────────────────────
st.markdown("# 🖥️ Simulasi Stokastik Sistem Antrian Server")
st.markdown(
    "**Pemodelan & Simulasi Stokastik** — Distribusi Poisson & Eksponensial | Model M/M/c Queue"
)
st.divider()

if not run:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:20px">M/M/c</div>
            <div class="metric-sub">Kendall's Notation</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-card {'metric-good' if rho < 0.7 else 'metric-warn' if rho < 1 else 'metric-bad'}">
            <div class="metric-label">Utilisasi ρ saat ini</div>
            <div class="metric-value">{rho:.2f}</div>
            <div class="metric-sub">{rho_status}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-label">Algoritma tersedia</div>
            <div class="metric-value" style="font-size:20px">3</div>
            <div class="metric-sub">RR · Random · LC</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-left:4px solid #64b5f6;border-radius:10px;padding:20px 24px;margin-bottom:16px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Cara Penggunaan</div>
        <div style="font-size:13px;color:#90a4ae;line-height:2.2">
        1. Atur <b style="color:#64b5f6">λ</b> — arrival rate request per detik (Distribusi Poisson)<br>
        2. Atur <b style="color:#64b5f6">μ</b> — service rate per worker (Distribusi Eksponensial)<br>
        3. Atur <b style="color:#64b5f6">c</b> — jumlah worker paralel<br>
        4. Pastikan <b style="color:#4caf50">ρ = λ/(c·μ) &lt; 1</b> agar sistem stabil<br>
        5. Pilih algoritma load balancing<br>
        6. Klik <b style="color:#64b5f6">▶ Jalankan Simulasi</b> di sidebar
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-radius:10px;padding:20px 24px;margin-bottom:16px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Rumus Utama</div>
        <div style="font-family:monospace;font-size:13px;color:#90caf9;line-height:2.2">
        ρ = λ / (c × μ) &nbsp;&nbsp;&nbsp; ← utilisasi sistem (harus &lt; 1)<br>
        L = λ × W &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ← Little's Law (rata-rata request dalam sistem)<br>
        W = Wq + 1/μ &nbsp;&nbsp;&nbsp;&nbsp; ← total time = wait + service
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-radius:10px;padding:20px 24px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Algoritma Load Balancing</div>
        <div style="font-size:13px;color:#90a4ae;line-height:2.2">
        🔄 <b style="color:#64b5f6">Round Robin</b> — request didistribusikan bergiliran, cocok untuk beban homogen<br>
        🎲 <b style="color:#64b5f6">Random</b> — worker dipilih acak, variansi tinggi, overhead nol<br>
        📉 <b style="color:#64b5f6">Least Connection</b> — selalu ke worker paling kosong, optimal untuk beban heterogen
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # st.markdown("<br>", unsafe_allow_html=True)
    # st.link_button("🎬 Buka Animasi Interaktif", "https://aufaanggara.github.io/server/dashboard/animation.html")
    st.markdown("### 🎬 Animasi Interaktif")
    st.components.v1.iframe(
        "https://aufaanggara.github.io/server/dashboard/animation.html",
        height=650,
        scrolling=False,
    )

else:
    # ── Run simulation ────────────────────────────────────────────
    with st.spinner("⏳ Menjalankan simulasi stokastik..."):
        results = run_simulation(lambda_rate, num_workers, algorithm, duration, mu)
        all_results = run_all_algorithms(lambda_rate, num_workers, duration, mu)

    df = pd.DataFrame(results)
    tput = len(df) / duration
    avg_wait = df["wait_time"].mean()
    avg_svc = df["service_time"].mean()
    avg_total = df["total_time"].mean()
    L = tput * avg_total
    rho = lambda_rate / (num_workers * mu)

    # ── Stability alert ───────────────────────────────────────────
    if rho >= 1:
        st.markdown(
            f'<div class="alert-critical">🔴 <b>SISTEM TIDAK STABIL</b> — ρ = {rho:.2f} ≥ 1. Antrian akan terus membesar tanpa batas. Tambah worker atau kurangi λ.</div>',
            unsafe_allow_html=True,
        )
    elif rho >= 0.8:
        st.markdown(
            f'<div class="alert-warn">⚠️ <b>Mendekati kritis</b> — ρ = {rho:.2f}. Sistem mulai jenuh, wait time meningkat signifikan.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-stable">✅ <b>Sistem stabil</b> — ρ = {rho:.2f}. Kapasitas server mencukupi untuk menangani beban.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "Total Request", len(df), "sejak reset", ""),
        (c2, "Throughput", f"{tput:.2f}", "req/s aktual", ""),
        (c3, "Avg Service Time", f"{avg_svc:.3f}s", "distribusi Exp(μ)", ""),
        (
            c4,
            "Avg Wait Time",
            f"{avg_wait:.3f}s",
            "Wq rata-rata",
            "warn" if avg_wait > 1 else "good",
        ),
        (
            c5,
            "Utilisasi ρ",
            f"{rho:.2f}",
            rho_status,
            "bad" if rho >= 1 else "warn" if rho >= 0.8 else "good",
        ),
        (c6, "Little's L", f"{L:.2f}", f"λ×W = {tput:.2f}×{avg_total:.2f}", ""),
    ]
    for col, label, value, sub, cls in metrics:
        col.markdown(
            f"""
        <div class="metric-card {'metric-'+cls if cls else ''}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        df["time_bucket"] = (df["arrive_time"] // 5) * 5
        tput_df = df.groupby("time_bucket").size().reset_index(name="count")
        fig1 = px.area(
            tput_df,
            x="time_bucket",
            y="count",
            labels={"time_bucket": "Waktu (s)", "count": "Jumlah Request"},
        )
        fig1.update_traces(line_color="#64b5f6", fillcolor="rgba(100,181,246,0.15)")
        plotly_dark(fig1, "📈 Request per 5 Detik")
        st.plotly_chart(fig1, use_container_width=True, key="fig1")

    with col_r:
        fig2 = px.histogram(
            df,
            x="service_time",
            nbins=25,
            labels={"service_time": "Service Time (s)", "count": "Frekuensi"},
        )
        fig2.update_traces(marker_color="#64b5f6", marker_opacity=0.7)

        # Overlay kurva Eksponensial teoritis
        x_range = np.linspace(0, df["service_time"].max(), 200)
        y_exp = mu * np.exp(-mu * x_range) * len(df) * (df["service_time"].max() / 25)
        fig2.add_trace(
            go.Scatter(
                x=x_range,
                y=y_exp,
                mode="lines",
                line=dict(color="#ef9f27", width=2, dash="dot"),
                name="Exp(μ) teoritis",
            )
        )
        plotly_dark(fig2, "📊 Distribusi Service Time + Kurva Teoritis")
        st.plotly_chart(fig2, use_container_width=True, key="fig2")

    # ── Charts row 2 ──────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        worker_df = df.groupby("worker_id").size().reset_index(name="requests")
        colors = [
            "#4caf50" if q == 0 else "#ff9800" if q <= 2 else "#f44336"
            for q in worker_df["requests"]
        ]
        fig3 = px.bar(
            worker_df,
            x="worker_id",
            y="requests",
            labels={"worker_id": "Worker ID", "requests": "Jumlah Request"},
        )
        fig3.update_traces(marker_color=colors)
        plotly_dark(
            fig3, f"👷 Request per Worker ({algorithm.replace('_',' ').title()})"
        )
        st.plotly_chart(fig3, use_container_width=True, key="fig3")

    with col_r2:
        # Scatter: arrive time vs wait time
        fig4 = px.scatter(
            df,
            x="arrive_time",
            y="wait_time",
            color="worker_id",
            labels={
                "arrive_time": "Waktu Kedatangan (s)",
                "wait_time": "Wait Time (s)",
            },
            color_continuous_scale="Blues",
        )
        plotly_dark(fig4, "⏱ Wait Time vs Waktu Kedatangan")
        st.plotly_chart(fig4, use_container_width=True, key="fig4")

    st.divider()

    # ── Utilisasi per interval ────────────────────────────────────
    df["interval"] = (df["arrive_time"] // 5) * 5
    util_df = (
        df.groupby("interval")
        .apply(lambda g: min(len(g) / (5 * lambda_rate), 1.5))
        .reset_index(name="utilisasi")
    )
    fig_util = px.area(
        util_df,
        x="interval",
        y="utilisasi",
        labels={"interval": "Waktu (s)", "utilisasi": "Utilisasi Estimasi"},
    )
    fig_util.update_traces(line_color="#ff9800", fillcolor="rgba(255,152,0,0.1)")
    fig_util.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="#f44336",
        annotation_text="ρ = 1 (batas kritis)",
        annotation_position="top right",
    )
    fig_util.add_hline(
        y=rho,
        line_dash="dash",
        line_color="#4caf50",
        annotation_text=f"ρ teoritis = {rho:.2f}",
        annotation_position="bottom right",
    )
    plotly_dark(fig_util, "📉 Estimasi Utilisasi per Interval Waktu")
    st.plotly_chart(fig_util, use_container_width=True, key="fig_util")

    st.divider()

    # ── Perbandingan algoritma ────────────────────────────────────
    st.markdown("### 📊 Perbandingan Algoritma Load Balancer")
    comparison = []
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            r = lambda_rate / (num_workers * mu)
            comparison.append(
                {
                    "Algoritma": algo.replace("_", " ").title(),
                    "Total Request": len(d),
                    "Throughput (req/s)": round(len(d) / duration, 3),
                    "Avg Service Time (s)": round(d["service_time"].mean(), 4),
                    "Avg Wait Time (s)": round(d["wait_time"].mean(), 4),
                    "Avg Total Time (s)": round(d["total_time"].mean(), 4),
                    "L = λW": round((len(d) / duration) * d["total_time"].mean(), 3),
                }
            )
    comp_df = pd.DataFrame(comparison)

    col_tbl, col_bar = st.columns([1.2, 1])
    with col_tbl:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    with col_bar:
        fig5 = px.bar(
            comp_df,
            x="Algoritma",
            y="Throughput (req/s)",
            color="Algoritma",
            color_discrete_sequence=["#64b5f6", "#4caf50", "#ff9800"],
        )
        plotly_dark(fig5, "Throughput per Algoritma")
        st.plotly_chart(fig5, use_container_width=True, key="fig5")

    # Wait time comparison
    fig6 = go.Figure()
    algo_colors = {
        "Round Robin": "#64b5f6",
        "Random": "#4caf50",
        "Least Connection": "#ff9800",
    }
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            fig6.add_trace(
                go.Box(
                    y=d["wait_time"],
                    name=algo.replace("_", " ").title(),
                    marker_color=algo_colors.get(
                        algo.replace("_", " ").title(), "#64b5f6"
                    ),
                    boxmean=True,
                )
            )
    plotly_dark(fig6, "📦 Distribusi Wait Time per Algoritma (Box Plot)")
    st.plotly_chart(fig6, use_container_width=True, key="fig6")

    st.divider()

    # ── Little's Law ──────────────────────────────────────────────
    st.markdown("### 📐 Validasi Little's Law")
    st.markdown(
        f"""
    <div class="littles-card">
        <div style="font-size:12px;color:#546e7a;margin-bottom:8px">L = λ × W</div>
        <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
        <div style="font-size:12px;color:#78909c;margin-top:8px">
        λ (arrival rate aktual) = <b style="color:#64b5f6">{tput:.3f} req/s</b> &nbsp;|&nbsp;
        W (avg total time) = <b style="color:#64b5f6">{avg_total:.3f}s</b> &nbsp;|&nbsp;
        L (rata-rata dalam sistem) = <b style="color:#64b5f6">{L:.3f} request</b>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Animasi link ──────────────────────────────────────────────
    st.markdown("### 🎬 Animasi Interaktif")
    st.markdown(
        """
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-left:4px solid #64b5f6;border-radius:10px;padding:16px 20px;margin-bottom:12px">
        <div style="font-size:13px;color:#90a4ae">
        Visualisasi packet flow secara real-time — paket data mengalir dari request generator ke load balancer ke worker, 
        dengan warna worker yang berubah sesuai beban (hijau → kuning → merah).
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🎬 Animasi Interaktif")
    st.components.v1.iframe(
        "https://aufaanggara.github.io/server/dashboard/animation.html",
        height=650,
        scrolling=False,
    )

    # ── Log tabel ─────────────────────────────────────────────────
    st.markdown("### 📋 Log Hasil Simulasi")
    col_log, col_dl = st.columns([4, 1])
    with col_log:
        st.dataframe(
            df[
                [
                    "request_id",
                    "worker_id",
                    "arrive_time",
                    "wait_time",
                    "service_time",
                    "total_time",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with col_dl:
        st.markdown("<br><br>", unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "simulation_results.csv", "text/csv")
        st.success("✅ Simulasi selesai!")
