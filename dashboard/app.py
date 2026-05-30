import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import httpx
import threading
import time
from datetime import datetime

st.set_page_config(
    page_title="Simulasi Antrian Server — Live",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; }
section[data-testid="stSidebar"] { background-color: #0d1422; border-right: 1px solid #1a2a3a; }
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
h1 { color: #64b5f6 !important; font-size: 1.6rem !important; letter-spacing: 0.5px; }
h2, h3 { color: #90caf9 !important; }
p, .stMarkdown { color: #b0bec5 !important; }
.metric-card {
    background: linear-gradient(135deg, #0d1a2e, #0f2040);
    border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 14px 18px; text-align: center; margin-bottom: 8px;
}
.metric-label { font-size: 10px; color: #546e7a; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 28px; font-weight: 700; color: #64b5f6; line-height: 1; }
.metric-sub { font-size: 11px; color: #78909c; margin-top: 4px; }
.metric-good .metric-value { color: #4caf50; }
.metric-warn .metric-value { color: #ff9800; }
.metric-bad .metric-value { color: #f44336; }
.rho-card {
    background: linear-gradient(135deg, #0d1a2e, #0f2040);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
    font-family: monospace; font-size: 13px; color: #90caf9;
    border-left: 3px solid #64b5f6;
}
.littles-card {
    background: linear-gradient(135deg, #0d1a2e 0%, #102030 100%);
    border: 1px solid #1e3a5f; border-left: 4px solid #64b5f6;
    border-radius: 10px; padding: 16px 20px; margin: 8px 0;
    font-size: 14px; color: #90caf9;
}
.littles-card .eq { font-size: 20px; font-weight: 700; color: #64b5f6; font-family: monospace; }
.alert-stable { background: #0d2318; border: 1px solid #2e7d32; border-radius: 8px; padding: 10px 16px; color: #a5d6a7; font-size: 13px; }
.alert-critical { background: #2d1010; border: 1px solid #c62828; border-radius: 8px; padding: 10px 16px; color: #ef9a9a; font-size: 13px; }
.alert-warn { background: #2d1f00; border: 1px solid #e65100; border-radius: 8px; padding: 10px 16px; color: #ffcc80; font-size: 13px; }
.server-card {
    background: #0d1422; border: 1px solid #1a2a3a; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 6px; font-size: 12px;
}
.server-online { border-left: 3px solid #4caf50; }
.server-offline { border-left: 3px solid #f44336; }
.section-header {
    font-size: 12px; color: #546e7a; text-transform: uppercase;
    letter-spacing: 1px; margin: 16px 0 8px; padding-bottom: 4px;
    border-bottom: 1px solid #1a2a3a;
}
.stButton > button {
    background: #1565c0; color: white; border: none;
    border-radius: 6px; font-weight: 600; width: 100%;
    padding: 10px; font-size: 14px;
}
.stButton > button:hover { background: #1976d2; }
hr { border-color: #1a2a3a !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────
SERVER_URL = "http://192.168.56.101:8000"

def plotly_dark(fig, title=""):
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1422",
        font_color="#b0bec5",
        title=dict(text=title, font=dict(color="#90caf9", size=13)),
        xaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        yaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="#0d1422", bordercolor="#1a2a3a"),
    )
    return fig

def check_server():
    try:
        r = httpx.get(f"{SERVER_URL}/", timeout=3.0)
        return r.status_code == 200, r.json()
    except:
        return False, {}

results_lock = threading.Lock()
live_results = []

def send_request(request_id, algo):
    try:
        start = time.time()
        r = httpx.post(
            f"{SERVER_URL}/process",
            params={"request_id": request_id, "algorithm": algo},
            timeout=15.0
        )
        elapsed = time.time() - start
        data = r.json()
        data["total_time"] = round(elapsed, 4)
        data["timestamp"] = datetime.now().isoformat()
        data["algorithm"] = algo
        with results_lock:
            live_results.append(data)
    except:
        pass

# ── Session state ─────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = []

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")

    # Server status
    st.markdown('<div class="section-header">Status Server VM</div>', unsafe_allow_html=True)
    server_ok, server_info = check_server()
    if server_ok:
        workers_active = server_info.get("workers", "?")
        st.markdown(f"""
        <div class="server-card server-online">
            <b style="color:#4caf50">● Server Online</b><br>
            <span style="color:#78909c">IP: 192.168.56.101:8000</span><br>
            <span style="color:#78909c">Workers aktif: {workers_active}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="server-card server-offline">
            <b style="color:#f44336">● Server Offline</b><br>
            <span style="color:#78909c">Pastikan VM Ubuntu menyala</span><br>
            <span style="color:#78909c">dan FastAPI berjalan di port 8000</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Parameter Stokastik</div>', unsafe_allow_html=True)
    lambda_rate = st.slider("λ — Arrival Rate (req/s)", 0.5, 5.0, 2.0, 0.5)
    num_workers = st.slider("Jumlah Worker (c)", 1, 5, 3)
    mu_display = st.slider("μ — Service Rate (display)", 0.5, 3.0, 1.0, 0.5)

    rho = lambda_rate / (num_workers * mu_display)
    rho_color = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_status = "✅ Stabil" if rho < 0.7 else "⚠️ Mendekati kritis" if rho < 1 else "🔴 Tidak stabil!"
    st.markdown(f"""
    <div class="rho-card">
        ρ = λ/(c·μ) = {lambda_rate:.1f}/({num_workers}×{mu_display:.1f}) =
        <span style="color:{rho_color};font-weight:700;font-size:16px">{rho:.2f}</span><br>
        <span style="color:{rho_color};font-size:12px">{rho_status}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Konfigurasi Simulasi</div>', unsafe_allow_html=True)
    algorithm = st.selectbox(
        "Algoritma Load Balancer",
        ["round_robin", "random", "least_connection"],
        format_func=lambda x: {"round_robin": "🔄 Round Robin", "random": "🎲 Random", "least_connection": "📉 Least Connection"}[x]
    )
    duration = st.slider("Durasi Simulasi (detik)", 10, 60, 30, 10)

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Jalankan"):
            if server_ok:
                st.session_state.running = True
                st.session_state.results = []
                try:
                    httpx.post(f"{SERVER_URL}/config", params={"num_workers": num_workers})
                    httpx.delete(f"{SERVER_URL}/reset")
                except:
                    pass
            else:
                st.error("Server offline!")
    with col_btn2:
        if st.button("⏹ Stop"):
            st.session_state.running = False

    st.markdown('<div class="section-header">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#546e7a;line-height:1.7">
    <b style="color:#78909c">M/M/c Queue (Live):</b><br>
    • Request dikirim ke VM nyata<br>
    • FastAPI memproses via HTTP<br>
    • Latency jaringan ikut terukur<br>
    • ρ &lt; 1 → sistem stabil
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────
st.markdown("# 🖥️ Simulasi Stokastik Sistem Antrian Server")
st.markdown("**Mode Live** — Request dikirim ke VM Ubuntu Server via jaringan Host-Only")
st.divider()

if not st.session_state.running and not st.session_state.results:
    # ── Landing ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card {'metric-good' if server_ok else 'metric-bad'}">
            <div class="metric-label">VM Server</div>
            <div class="metric-value" style="font-size:20px">{'ON' if server_ok else 'OFF'}</div>
            <div class="metric-sub">{'192.168.56.101:8000' if server_ok else 'Tidak terdeteksi'}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card {'metric-good' if rho < 0.7 else 'metric-warn' if rho < 1 else 'metric-bad'}">
            <div class="metric-label">Utilisasi ρ</div>
            <div class="metric-value">{rho:.2f}</div>
            <div class="metric-sub">{rho_status}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Mode</div>
            <div class="metric-value" style="font-size:18px">Live VM</div>
            <div class="metric-sub">Request nyata via HTTP</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not server_ok:
        st.markdown("""
        <div class="alert-critical">
        🔴 <b>Server VM tidak terdeteksi.</b> Pastikan:<br>
        1. VM Ubuntu Server menyala di VirtualBox<br>
        2. FastAPI berjalan: <code>uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4</code><br>
        3. IP VM adalah 192.168.56.101
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-stable">
        ✅ Server VM online dan siap menerima request. Atur parameter di sidebar lalu klik <b>▶ Jalankan</b>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.link_button("🎬 Buka Animasi Interaktif", "https://aufaanggara.github.io/server/dashboard/animation.html")

elif st.session_state.running:
    # ── Live simulation ───────────────────────────────────────────
    live_results.clear()
    start_time = time.time()
    request_id = 0
    threads = []

    col1, col2, col3, col4 = st.columns(4)
    metric_req = col1.empty()
    metric_tput = col2.empty()
    metric_svc = col3.empty()
    metric_wait = col4.empty()

    chart_ph = st.empty()
    worker_ph = st.empty()
    littles_ph = st.empty()
    table_ph = st.empty()

    while time.time() - start_time < duration and st.session_state.running:
        inter_arrival = np.random.exponential(1.0 / lambda_rate)
        time.sleep(inter_arrival)

        t = threading.Thread(target=send_request, args=(request_id, algorithm))
        t.start()
        threads.append(t)
        request_id += 1

        if request_id % 3 == 0:
            with results_lock:
                current = live_results.copy()

            if current:
                df = pd.DataFrame(current)
                elapsed = time.time() - start_time
                tput = len(df) / elapsed
                avg_svc = df["service_time"].mean()
                avg_wait = df.get("wait_time", pd.Series([0])).mean() if "wait_time" in df.columns else 0

                for col, label, value, sub, cls in [
                    (metric_req, "Total Request", len(df), "sejak mulai", ""),
                    (metric_tput, "Throughput", f"{tput:.2f}", "req/s aktual", ""),
                    (metric_svc, "Avg Service", f"{avg_svc:.3f}s", "dari VM", ""),
                    (metric_wait, "Utilisasi ρ", f"{rho:.2f}", rho_status, "bad" if rho >= 1 else "warn" if rho >= 0.8 else "good"),
                ]:
                    col.markdown(f"""
                    <div class="metric-card {'metric-'+cls if cls else ''}">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grafik throughput
                df["time_bucket"] = (df.index // 3) * 3
                tput_df = df.groupby("time_bucket").size().reset_index(name="count")
                fig1 = px.area(tput_df, x="time_bucket", y="count",
                    labels={"time_bucket": "Request ke-", "count": "Jumlah"})
                fig1.update_traces(line_color="#64b5f6", fillcolor="rgba(100,181,246,0.15)")
                plotly_dark(fig1, "📈 Request Masuk (Live)")
                chart_ph.plotly_chart(fig1, use_container_width=True, key=f"live_chart_{request_id}")

                # Beban per worker
                if "worker_id" in df.columns:
                    worker_df = df.groupby("worker_id").size().reset_index(name="requests")
                    fig_w = px.bar(worker_df, x="worker_id", y="requests",
                        labels={"worker_id": "Worker", "requests": "Request"})
                    fig_w.update_traces(marker_color="#64b5f6")
                    plotly_dark(fig_w, "👷 Distribusi per Worker (Live)")
                    worker_ph.plotly_chart(fig_w, use_container_width=True, key=f"live_worker_{request_id}")

                # Little's Law
                avg_total = df["total_time"].mean()
                L = tput * avg_total
                littles_ph.markdown(f"""
                <div class="littles-card">
                    <div style="font-size:12px;color:#546e7a;margin-bottom:6px">Little's Law — L = λW</div>
                    <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

                # Tabel live
                table_ph.dataframe(
                    df[["request_id", "worker_id", "service_time", "total_time", "timestamp"]].tail(10),
                    use_container_width=True
                )

    for t in threads:
        t.join()

    st.session_state.running = False
    with results_lock:
        st.session_state.results = live_results.copy()
    st.rerun()

else:
    # ── Hasil simulasi ────────────────────────────────────────────
    df = pd.DataFrame(st.session_state.results)

    if df.empty:
        st.warning("Tidak ada data hasil simulasi.")
    else:
        tput = len(df) / duration
        avg_svc = df["service_time"].mean()
        avg_total = df["total_time"].mean()
        avg_wait = df["wait_time"].mean() if "wait_time" in df.columns else 0
        L = tput * avg_total

        if rho >= 1:
            st.markdown(f'<div class="alert-critical">🔴 <b>SISTEM TIDAK STABIL</b> — ρ = {rho:.2f} ≥ 1.</div>', unsafe_allow_html=True)
        elif rho >= 0.8:
            st.markdown(f'<div class="alert-warn">⚠️ <b>Mendekati kritis</b> — ρ = {rho:.2f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-stable">✅ <b>Sistem stabil</b> — ρ = {rho:.2f}.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, label, value, sub, cls in [
            (c1, "Total Request", len(df), "berhasil dikirim", ""),
            (c2, "Throughput", f"{tput:.2f}", "req/s aktual", ""),
            (c3, "Avg Service", f"{avg_svc:.3f}s", "dari VM nyata", ""),
            (c4, "Avg Total Time", f"{avg_total:.3f}s", "inkl. latency", ""),
            (c5, "Utilisasi ρ", f"{rho:.2f}", rho_status, "bad" if rho >= 1 else "warn" if rho >= 0.8 else "good"),
            (c6, "Little's L", f"{L:.2f}", f"{tput:.2f}×{avg_total:.2f}", ""),
        ]:
            col.markdown(f"""
            <div class="metric-card {'metric-'+cls if cls else ''}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            df["time_bucket"] = (df.index // 5) * 5
            tput_df = df.groupby("time_bucket").size().reset_index(name="count")
            fig1 = px.area(tput_df, x="time_bucket", y="count",
                labels={"time_bucket": "Request ke-", "count": "Jumlah"})
            fig1.update_traces(line_color="#64b5f6", fillcolor="rgba(100,181,246,0.15)")
            plotly_dark(fig1, "📈 Request per Interval")
            st.plotly_chart(fig1, use_container_width=True, key="res_fig1")

        with col_r:
            fig2 = px.histogram(df, x="service_time", nbins=20,
                labels={"service_time": "Service Time (s)"})
            fig2.update_traces(marker_color="#64b5f6", marker_opacity=0.7)
            x_range = np.linspace(0, df["service_time"].max(), 200)
            y_exp = np.exp(-x_range) * len(df) * (df["service_time"].max() / 20)
            fig2.add_trace(go.Scatter(x=x_range, y=y_exp, mode="lines",
                line=dict(color="#ef9f27", width=2, dash="dot"), name="Exp teoritis"))
            plotly_dark(fig2, "📊 Distribusi Service Time")
            st.plotly_chart(fig2, use_container_width=True, key="res_fig2")

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            if "worker_id" in df.columns:
                worker_df = df.groupby("worker_id").size().reset_index(name="requests")
                fig3 = px.bar(worker_df, x="worker_id", y="requests",
                    labels={"worker_id": "Worker", "requests": "Request"}, color="worker_id")
                plotly_dark(fig3, f"👷 Request per Worker ({algorithm.replace('_',' ').title()})")
                st.plotly_chart(fig3, use_container_width=True, key="res_fig3")

        with col_r2:
            fig4 = px.scatter(df, x=df.index, y="total_time", color="worker_id",
                labels={"x": "Request ke-", "total_time": "Total Time (s)"},
                color_continuous_scale="Blues")
            plotly_dark(fig4, "⏱ Total Time per Request")
            st.plotly_chart(fig4, use_container_width=True, key="res_fig4")

        st.divider()

        st.markdown("### 📐 Validasi Little's Law")
        st.markdown(f"""
        <div class="littles-card">
            <div style="font-size:12px;color:#546e7a;margin-bottom:8px">L = λ × W</div>
            <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
            <div style="font-size:12px;color:#78909c;margin-top:8px">
            λ = <b style="color:#64b5f6">{tput:.3f} req/s</b> &nbsp;|&nbsp;
            W = <b style="color:#64b5f6">{avg_total:.3f}s</b> &nbsp;|&nbsp;
            L = <b style="color:#64b5f6">{L:.3f} request dalam sistem</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🎬 Animasi Interaktif")
        st.markdown("""
        <div style="background:#0d1422;border:1px solid #1a2a3a;border-left:4px solid #64b5f6;border-radius:10px;padding:16px 20px;margin-bottom:12px">
            <div style="font-size:13px;color:#90a4ae">
            Visualisasi packet flow real-time — cocok untuk ditampilkan saat presentasi di kelas berdampingan dengan dashboard ini.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("🎬 Buka Animasi di Tab Baru", "https://aufaanggara.github.io/server/dashboard/animation.html")

        st.divider()

        st.markdown("### 📋 Log Hasil Simulasi")
        col_log, col_dl = st.columns([4, 1])
        with col_log:
            st.dataframe(
                df[["request_id", "worker_id", "service_time", "total_time", "timestamp"]],
                use_container_width=True, hide_index=True
            )
        with col_dl:
            st.markdown("<br><br>", unsafe_allow_html=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "simulation_results_live.csv", "text/csv")
            st.success("✅ Simulasi selesai!")