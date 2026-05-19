import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import httpx
import threading
import time
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="Simulasi Antrian Server", page_icon="🖥️", layout="wide")

SERVER_URL = "http://192.168.56.101:8000"

# Session state
if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = []
if "log" not in st.session_state:
    st.session_state.log = []

# ── SIDEBAR ──────────────────────────────────────────
st.sidebar.title("⚙️ Panel Kontrol")

lambda_rate = st.sidebar.slider("λ — Request per detik", 0.5, 5.0, 2.0, 0.5)
num_workers = st.sidebar.slider("Jumlah Worker", 1, 5, 3)
algorithm = st.sidebar.selectbox(
    "Algoritma Load Balancer", ["round_robin", "random", "least_connection"]
)
duration = st.sidebar.slider("Durasi Simulasi (detik)", 10, 60, 30, 10)

if st.sidebar.button("▶ Jalankan Simulasi"):
    st.session_state.running = True
    st.session_state.results = []
    st.session_state.log = []

    # Set jumlah worker di server
    try:
        httpx.post(f"{SERVER_URL}/config", params={"num_workers": num_workers})
        httpx.delete(f"{SERVER_URL}/reset")
    except:
        st.sidebar.error("Gagal koneksi ke server!")
        st.session_state.running = False

if st.sidebar.button("⏹ Stop"):
    st.session_state.running = False

# ── HEADER ───────────────────────────────────────────
st.title("🖥️ Simulasi Stokastik Sistem Antrian Server")
st.markdown("**Pemodelan & Simulasi Stokastik** — Distribusi Poisson & Eksponensial")
st.divider()

# ── LAYOUT UTAMA ─────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
metric_requests = col1.metric("Total Request", 0)
metric_throughput = col2.metric("Throughput", "0 req/s")
metric_avg_service = col3.metric("Avg Service Time", "0s")
metric_avg_total = col4.metric("Avg Total Time", "0s")

st.divider()

# Grafik
col_left, col_right = st.columns(2)
chart_throughput = col_left.empty()
chart_service = col_right.empty()

col_left2, col_right2 = st.columns(2)
chart_workers = col_left2.empty()
chart_comparison = col_right2.empty()

st.divider()

# Little's Law
st.subheader("📐 Validasi Little's Law (L = λW)")
littles_placeholder = st.empty()

st.divider()

# Tabel hasil
st.subheader("📋 Log Hasil Simulasi")
table_placeholder = st.empty()

# ── SIMULASI REAL-TIME ────────────────────────────────
results_lock = threading.Lock()
live_results = []


def send_request(request_id, algo):
    try:
        start = time.time()
        r = httpx.post(
            f"{SERVER_URL}/process",
            params={"request_id": request_id, "algorithm": algo},
            timeout=15.0,
        )
        elapsed = time.time() - start
        data = r.json()
        data["total_time"] = round(elapsed, 4)
        data["timestamp"] = datetime.now().isoformat()
        data["algorithm"] = algo
        with results_lock:
            live_results.append(data)
    except Exception as e:
        pass


if st.session_state.running:
    live_results.clear()
    start_time = time.time()
    request_id = 0
    threads = []

    while time.time() - start_time < duration and st.session_state.running:
        inter_arrival = np.random.exponential(1.0 / lambda_rate)
        time.sleep(inter_arrival)

        t = threading.Thread(target=send_request, args=(request_id, algorithm))
        t.start()
        threads.append(t)
        request_id += 1

        # Update dashboard setiap 3 request
        if request_id % 3 == 0:
            with results_lock:
                current = live_results.copy()

            if current:
                df = pd.DataFrame(current)

                # Update metrics
                avg_svc = df["service_time"].mean()
                avg_tot = df["total_time"].mean()
                tput = len(df) / (time.time() - start_time)

                col1.metric("Total Request", len(df))
                col2.metric("Throughput", f"{tput:.2f} req/s")
                col3.metric("Avg Service Time", f"{avg_svc:.3f}s")
                col4.metric("Avg Total Time", f"{avg_tot:.3f}s")

                # Grafik throughput per waktu
                df["time_bucket"] = pd.to_datetime(df["timestamp"]).dt.floor("5s")
                tput_df = df.groupby("time_bucket").size().reset_index(name="count")
                fig1 = px.line(
                    tput_df,
                    x="time_bucket",
                    y="count",
                    title="Request per 5 Detik",
                    labels={"time_bucket": "Waktu", "count": "Jumlah Request"},
                )
                chart_throughput.plotly_chart(
                    fig1, use_container_width=True, key=f"fig1_{request_id}"
                )

                # Grafik service time
                fig2 = px.histogram(
                    df,
                    x="service_time",
                    nbins=20,
                    title="Distribusi Service Time",
                    labels={"service_time": "Service Time (s)"},
                )
                chart_service.plotly_chart(
                    fig2, use_container_width=True, key=f"fig2_{request_id}"
                )

                # Grafik beban per worker
                worker_df = df.groupby("worker_id").size().reset_index(name="requests")
                fig3 = px.bar(
                    worker_df,
                    x="worker_id",
                    y="requests",
                    title="Request per Worker",
                    labels={"worker_id": "Worker", "requests": "Jumlah Request"},
                    color="worker_id",
                )
                chart_workers.plotly_chart(
                    fig3, use_container_width=True, key=f"fig3_{request_id}"
                )

                # Little's Law
                L = tput * avg_tot
                littles_placeholder.info(
                    f"λ (arrival rate) = **{tput:.3f} req/s** | "
                    f"W (avg time) = **{avg_tot:.3f}s** | "
                    f"L = λW = **{L:.3f}** request dalam sistem"
                )

                # Tabel
                table_placeholder.dataframe(
                    df[
                        [
                            "request_id",
                            "worker_id",
                            "service_time",
                            "total_time",
                            "timestamp",
                        ]
                    ].tail(20),
                    use_container_width=True,
                )

    for t in threads:
        t.join()

    st.session_state.running = False
    st.success("✅ Simulasi selesai!")

    # Simpan hasil
    if live_results:
        df_final = pd.DataFrame(live_results)
        df_final.to_csv("simulation_results.csv", index=False)
        st.session_state.results = live_results

# Tampilkan hasil lama kalau ada
elif st.session_state.results:
    df = pd.DataFrame(st.session_state.results)
    table_placeholder.dataframe(
        df[["request_id", "worker_id", "service_time", "total_time", "timestamp"]].tail(
            20
        ),
        use_container_width=True,
    )
