from fastapi import FastAPI
import time
import random

app = FastAPI()

@app.get("/")
def root():
    return {"status": "server running"}

@app.post("/process")
def process_request(worker_id: int = 0):
    service_time = random.expovariate(1.0)
    time.sleep(min(service_time, 2.0))
    return {
        "worker_id": worker_id,
        "service_time": round(service_time, 4),
        "status": "processed"
    }