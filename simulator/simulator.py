import httpx
import numpy as np
import time
import threading
import json
from datetime import datetime

# Konfigurasi
SERVER_URL = "http://192.168.56.101:8000"
LAMBDA = 2.0  # rata-rata request per detik (Poisson)
DURATION = 30  # durasi simulasi dalam detik
ALGORITHM = "round_robin"  # round_robin, random, least_connection

# Storage hasil
results = []
results_lock = threading.Lock()

def send_request(request_id: int):
    try:
        start = time.time()
        response = httpx.post(
            f"{SERVER_URL}/process",
            params={"request_id": request_id, "algorithm": ALGORITHM},
            timeout=10.0
        )
        elapsed = time.time() - start
        data = response.json()
        data["total_time"] = round(elapsed, 4)
        data["arrival_time"] = datetime.now().isoformat()

        with results_lock:
            results.append(data)
            print(f"[{request_id}] worker={data['worker_id']} "
                  f"service={data['service_time']}s "
                  f"total={data['total_time']}s")
    except Exception as e:
        print(f"[{request_id}] ERROR: {e}")

def run_simulation():
    print(f"Simulasi dimulai | λ={LAMBDA} req/s | durasi={DURATION}s | algoritma={ALGORITHM}")
    print("-" * 60)

    start_time = time.time()
    request_id = 0
    threads = []

    while time.time() - start_time < DURATION:
        # Generate inter-arrival time dengan distribusi Poisson
        inter_arrival = np.random.exponential(1.0 / LAMBDA)
        time.sleep(inter_arrival)

        t = threading.Thread(target=send_request, args=(request_id,))
        t.start()
        threads.append(t)
        request_id += 1

    # Tunggu semua thread selesai
    for t in threads:
        t.join()

    print("-" * 60)
    print(f"Simulasi selesai | Total request: {len(results)}")

    if results:
        avg_service = np.mean([r["service_time"] for r in results])
        avg_total = np.mean([r["total_time"] for r in results])
        print(f"Rata-rata service time: {round(avg_service, 4)}s")
        print(f"Rata-rata total time  : {round(avg_total, 4)}s")
        print(f"Throughput            : {len(results) / DURATION:.2f} req/s")

if __name__ == "__main__":
    run_simulation()