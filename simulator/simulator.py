import httpx
import numpy as np
import time
import threading
import pandas as pd
from datetime import datetime

# Konfigurasi
SERVER_URL = "http://192.168.56.101:8000"
LAMBDA = 2.0        # rata-rata request per detik
DURATION = 30       # durasi simulasi per algoritma
ALGORITHMS = ["round_robin", "random", "least_connection"]

results_lock = threading.Lock()

def send_request(request_id: int, algorithm: str, results: list):
    try:
        start = time.time()
        response = httpx.post(
            f"{SERVER_URL}/process",
            params={"request_id": request_id, "algorithm": algorithm},
            timeout=15.0
        )
        elapsed = time.time() - start
        data = response.json()
        data["total_time"] = round(elapsed, 4)
        data["algorithm"] = algorithm

        with results_lock:
            results.append(data)
            print(f"  [{algorithm}] req={request_id} "
                  f"worker={data['worker_id']} "
                  f"service={data['service_time']}s")
    except Exception as e:
        print(f"  [{algorithm}] req={request_id} ERROR: {e}")

def run_one_algorithm(algorithm: str) -> list:
    print(f"\nMenjalankan algoritma: {algorithm.upper()}")
    print("-" * 50)

    # Reset stats server dulu
    httpx.delete(f"{SERVER_URL}/reset")

    results = []
    start_time = time.time()
    request_id = 0
    threads = []

    while time.time() - start_time < DURATION:
        inter_arrival = np.random.exponential(1.0 / LAMBDA)
        time.sleep(inter_arrival)
        t = threading.Thread(
            target=send_request,
            args=(request_id, algorithm, results)
        )
        t.start()
        threads.append(t)
        request_id += 1

    for t in threads:
        t.join()

    return results

def print_comparison(all_results: dict):
    print("\n" + "=" * 60)
    print("PERBANDINGAN ALGORITMA LOAD BALANCER")
    print("=" * 60)
    print(f"{'Algoritma':<20} {'Total Req':<12} {'Avg Service':<14} {'Avg Total':<12} {'Throughput'}")
    print("-" * 60)

    for algo, results in all_results.items():
        if not results:
            continue
        avg_service = np.mean([r["service_time"] for r in results])
        avg_total = np.mean([r["total_time"] for r in results])
        throughput = len(results) / DURATION
        print(f"{algo:<20} {len(results):<12} {avg_service:<14.4f} {avg_total:<12.4f} {throughput:.2f} req/s")

    print("=" * 60)

    # Simpan ke CSV
    all_rows = []
    for algo, results in all_results.items():
        for r in results:
            all_rows.append(r)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv("simulation_results.csv", index=False)
        print("\nHasil disimpan ke simulation_results.csv")

if __name__ == "__main__":
    print("SIMULASI STOKASTIK SISTEM ANTRIAN SERVER")
    print(f"λ={LAMBDA} req/s | durasi={DURATION}s per algoritma")

    all_results = {}
    for algo in ALGORITHMS:
        all_results[algo] = run_one_algorithm(algo)

    print_comparison(all_results)