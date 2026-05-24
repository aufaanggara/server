import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Simulasi Antrian Server", page_icon="🖥️", layout="wide")

# ── SIDEBAR ──────────────────────────────────────────
st.sidebar.title("⚙️ Panel Kontrol")

lambda_rate = st.sidebar.slider("λ — Request per detik", 0.5, 5.0, 2.0, 0.5)
num_workers = st.sidebar.slider("Jumlah Worker", 1, 5, 3)
algorithm = st.sidebar.selectbox(
    "Algoritma Load Balancer", ["round_robin", "random", "least_connection"]
)
duration = st.sidebar.slider("Durasi Simulasi (detik)", 10, 60, 30, 10)
mu = st.sidebar.slider("μ — Service Rate per Worker", 0.5, 3.0, 1.0, 0.5)

run = st.sidebar.button("▶ Jalankan Simulasi")

# ── HEADER ───────────────────────────────────────────
st.title("🖥️ Simulasi Stokastik Sistem Antrian Server")
st.markdown("**Pemodelan & Simulasi Stokastik** — Distribusi Poisson & Eksponensial")
st.divider()


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

        # Hitung wait time
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

    # Kurangi queue setelah selesai
    for r in results:
        workers_queue[r["worker_id"]] = max(0, workers_queue[r["worker_id"]] - 1)

    return results


def run_all_algorithms(lambda_rate, num_workers, duration, mu):
    algos = ["round_robin", "random", "least_connection"]
    all_results = {}
    for algo in algos:
        all_results[algo] = run_simulation(lambda_rate, num_workers, algo, duration, mu)
    return all_results


if run:
    with st.spinner("Menjalankan simulasi..."):
        results = run_simulation(lambda_rate, num_workers, algorithm, duration, mu)
        all_results = run_all_algorithms(lambda_rate, num_workers, duration, mu)

    df = pd.DataFrame(results)

    # ── METRICS ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    tput = len(df) / duration
    col1.metric("Total Request", len(df))
    col2.metric("Throughput", f"{tput:.2f} req/s")
    col3.metric("Avg Service Time", f"{df['service_time'].mean():.3f}s")
    col4.metric("Avg Wait Time", f"{df['wait_time'].mean():.3f}s")

    st.divider()

    # ── GRAFIK ───────────────────────────────────────
    col_l, col_r = st.columns(2)

    # Arrival timeline
    df["time_bucket"] = (df["arrive_time"] // 5) * 5
    tput_df = df.groupby("time_bucket").size().reset_index(name="count")
    fig1 = px.line(
        tput_df,
        x="time_bucket",
        y="count",
        title="Request per 5 Detik",
        labels={"time_bucket": "Waktu (s)", "count": "Jumlah Request"},
    )
    col_l.plotly_chart(fig1, use_container_width=True, key="fig1")

    # Distribusi service time
    fig2 = px.histogram(
        df,
        x="service_time",
        nbins=20,
        title="Distribusi Service Time (Eksponensial)",
        labels={"service_time": "Service Time (s)"},
    )
    col_r.plotly_chart(fig2, use_container_width=True, key="fig2")

    col_l2, col_r2 = st.columns(2)

    # Beban per worker
    worker_df = df.groupby("worker_id").size().reset_index(name="requests")
    fig3 = px.bar(
        worker_df,
        x="worker_id",
        y="requests",
        title=f"Request per Worker ({algorithm})",
        labels={"worker_id": "Worker ID", "requests": "Jumlah Request"},
        color="worker_id",
    )
    col_l2.plotly_chart(fig3, use_container_width=True, key="fig3")

    # Wait time per worker
    wait_df = df.groupby("worker_id")["wait_time"].mean().reset_index()
    fig4 = px.bar(
        wait_df,
        x="worker_id",
        y="wait_time",
        title="Rata-rata Wait Time per Worker",
        labels={"worker_id": "Worker ID", "wait_time": "Avg Wait Time (s)"},
        color="worker_id",
    )
    col_r2.plotly_chart(fig4, use_container_width=True, key="fig4")

    st.divider()

    # ── PERBANDINGAN ALGORITMA ────────────────────────
    st.subheader("📊 Perbandingan Algoritma Load Balancer")
    comparison = []
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            comparison.append(
                {
                    "Algoritma": algo,
                    "Total Request": len(d),
                    "Throughput (req/s)": round(len(d) / duration, 3),
                    "Avg Service Time (s)": round(d["service_time"].mean(), 4),
                    "Avg Wait Time (s)": round(d["wait_time"].mean(), 4),
                    "Avg Total Time (s)": round(d["total_time"].mean(), 4),
                }
            )
    comp_df = pd.DataFrame(comparison)
    st.dataframe(comp_df, use_container_width=True)

    fig5 = px.bar(
        comp_df,
        x="Algoritma",
        y="Throughput (req/s)",
        title="Perbandingan Throughput per Algoritma",
        color="Algoritma",
    )
    st.plotly_chart(fig5, use_container_width=True, key="fig5")

    st.divider()

    # ── LITTLE'S LAW ─────────────────────────────────
    st.subheader("📐 Validasi Little's Law (L = λW)")
    avg_total = df["total_time"].mean()
    L = tput * avg_total
    st.info(
        f"λ (arrival rate) = **{tput:.3f} req/s** | "
        f"W (avg total time) = **{avg_total:.3f}s** | "
        f"L = λW = **{L:.3f}** request dalam sistem"
    )

    st.divider()

    # ── LOG TABEL ────────────────────────────────────
    st.subheader("📋 Log Hasil Simulasi")
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
    )

    # Simpan CSV
    df.to_csv("simulation_results.csv", index=False)
    st.success("✅ Simulasi selesai! Data tersimpan ke simulation_results.csv")

else:
    st.info("👈 Atur parameter di sidebar, lalu klik **▶ Jalankan Simulasi**")
