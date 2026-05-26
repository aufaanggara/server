# 🖥️ Simulasi Stokastik Sistem Antrian Server

Simulasi sistem antrian server dengan load balancing berbasis distribusi Poisson dan Eksponensial (model M/M/c). Server backend berjalan di VM Ubuntu Server via VirtualBox, request dikirim secara stokastik dari simulator Python, dan hasilnya divisualisasikan di dashboard Streamlit interaktif serta animasi p5.js.

**Mata Kuliah:** Teknik Pemodelan dan Simulasi Stokastik

**Demo:**
- 🚀 Dashboard: [serverr.streamlit.app](https://serverr.streamlit.app)
- 🎬 Animasi: [aufaanggara.github.io/server/dashboard/animation.html](https://aufaanggara.github.io/server/dashboard/animation.html)

---

## 📁 Struktur Folder

```
server/
├── server/
│   ├── main.py              # FastAPI backend (dijalankan di VM Ubuntu)
│   └── requirements.txt     # Dependencies untuk VM
├── simulator/
│   └── simulator.py         # Request generator stokastik (Poisson)
├── dashboard/
│   ├── app.py               # Dashboard Streamlit lokal (terhubung ke VM)
│   ├── app_cloud.py         # Dashboard Streamlit cloud (mandiri)
│   └── animation.html       # Animasi packet flow p5.js
├── docs/                    # Dokumentasi tambahan
├── requirements.txt         # Dependencies untuk Streamlit Cloud
└── README.md
```

---

## 🏗️ Arsitektur Sistem

```
[Laptop Host]                    [VM Ubuntu Server]
┌─────────────────┐    HTTP      ┌──────────────────────┐
│  simulator.py   │─────────────▶│  FastAPI (port 8000) │
│  (Poisson λ)    │              │  ┌────────────────┐   │
│                 │              │  │ Load Balancer  │   │
│  app.py         │◀─────────────│  │ Round Robin    │   │
│  (Streamlit)    │   Response   │  │ Random         │   │
└─────────────────┘              │  │ Least Conn.    │   │
                                 │  └────────────────┘   │
         Host-Only Network       │  Worker 0, 1, 2...5   │
         192.168.56.101:8000     └──────────────────────┘
```

---

## ⚙️ Setup & Instalasi

### Prasyarat

- VirtualBox (untuk VM Ubuntu Server)
- Python 3.10+ di laptop host
- Git

### 1. Clone Repository

```bash
git clone https://github.com/aufaanggara/server.git
cd server
```

### 2. Install Dependencies di Laptop Host

```bash
pip install streamlit numpy scipy httpx plotly pandas
```

### 3. Setup VM Ubuntu Server

**Buat VM di VirtualBox:**
- OS: Ubuntu Server 24.04 LTS
- RAM: 4096 MB, CPU: 2, Storage: 25 GB
- Network Adapter 1: NAT
- Network Adapter 2: Host-Only Adapter

**Di dalam VM (via SSH):**

```bash
# Install dependencies
pip install fastapi uvicorn simpy numpy --break-system-packages

# Tambahkan PATH
echo 'export PATH=$PATH:/home/aufa/.local/bin' >> ~/.bashrc && source ~/.bashrc

# Buka port 8000
sudo ufw allow 8000

# Cek IP Host-Only
ip addr show  # catat IP interface enp0s8, biasanya 192.168.56.x
```

**Copy kode server ke VM:**

```bash
# Di laptop host
ssh aufa@192.168.56.101
mkdir ~/server && cd ~/server
nano main.py  # paste isi server/main.py
```

**Jalankan FastAPI di VM:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Verifikasi server berjalan:**

Buka browser di laptop host: `http://192.168.56.101:8000` → harus muncul `{"status":"server running"}`

---

## 🚀 Cara Penggunaan

### Mode 1 — Jalankan Simulasi via Terminal

```bash
# Pastikan VM sudah menyala dan FastAPI berjalan
cd server
python simulator/simulator.py
```

Output:
```
SIMULASI STOKASTIK SISTEM ANTRIAN SERVER
λ=2.0 req/s | durasi=30s per algoritma

Menjalankan algoritma: ROUND_ROBIN
--------------------------------------------------
  [round_robin] req=0 worker=0 service=0.983s
  [round_robin] req=1 worker=1 service=0.622s
  ...

PERBANDINGAN ALGORITMA LOAD BALANCER
============================================================
Algoritma            Total Req    Avg Service    Throughput
round_robin          61           1.038          2.03 req/s
random               71           0.870          2.37 req/s
least_connection     48           1.339          1.60 req/s
============================================================
```

Konfigurasi simulator dapat diubah di `simulator/simulator.py`:

```python
LAMBDA = 2.0      # rata-rata request per detik
DURATION = 30     # durasi simulasi dalam detik
ALGORITHM = "round_robin"  # round_robin | random | least_connection
```

### Mode 2 — Dashboard Lokal (terhubung ke VM)

```bash
# Pastikan VM sudah menyala
streamlit run dashboard/app.py
```

Buka browser: `http://localhost:8501`

### Mode 3 — Dashboard Cloud (tanpa VM)

```bash
streamlit run dashboard/app_cloud.py
```

Atau akses langsung: [serverr.streamlit.app](https://serverr.streamlit.app)

### Mode 4 — Animasi p5.js

Buka file `dashboard/animation.html` di browser, atau akses:
[aufaanggara.github.io/server/dashboard/animation.html](https://aufaanggara.github.io/server/dashboard/animation.html)

---

## 🔬 Model Stokastik

### Distribusi yang Digunakan

| Komponen | Distribusi | Parameter | Implementasi |
|---|---|---|---|
| Kedatangan request | Poisson | λ (req/s) | `np.random.exponential(1/λ)` |
| Waktu pelayanan | Eksponensial | μ (req/s) | `np.random.exponential(1/μ)` |

### Algoritma Load Balancing

| Algoritma | Cara Kerja | Cocok Untuk |
|---|---|---|
| Round Robin | Bergiliran deterministik | Beban homogen, traffic stabil |
| Random | Acak uniform | Prototype, testing |
| Least Connection | Worker dengan antrian terpendek | Beban heterogen, service time bervariasi |

### Syarat Stabilitas

```
ρ = λ / (c × μ) < 1

Contoh: λ=2, c=3, μ=1 → ρ = 2/(3×1) = 0.67 ✅ Stabil
        λ=4, c=3, μ=1 → ρ = 4/(3×1) = 1.33 ❌ Tidak stabil
```

### Validasi — Little's Law

```
L = λ × W

L = rata-rata jumlah request dalam sistem
λ = arrival rate
W = rata-rata waktu total dalam sistem (wait + service)
```

---

## 📊 Endpoint API (FastAPI)

| Method | Endpoint | Deskripsi |
|---|---|---|
| GET | `/` | Status server dan jumlah worker aktif |
| POST | `/process` | Proses satu request (params: request_id, algorithm) |
| GET | `/stats` | Statistik real-time (throughput, avg wait time, dll) |
| POST | `/config` | Ubah jumlah worker aktif (params: num_workers) |
| DELETE | `/reset` | Reset semua log dan statistik |

**Contoh request:**

```bash
# Cek status
curl http://192.168.56.101:8000/

# Kirim request dengan algoritma round robin
curl -X POST "http://192.168.56.101:8000/process?request_id=1&algorithm=round_robin"

# Lihat statistik
curl http://192.168.56.101:8000/stats

# Ubah ke 5 worker
curl -X POST "http://192.168.56.101:8000/config?num_workers=5"

# Reset statistik
curl -X DELETE http://192.168.56.101:8000/reset
```

---

## 🛠️ Tech Stack

| Layer | Teknologi | Fungsi |
|---|---|---|
| Backend | FastAPI + Uvicorn | Server penerima request di VM |
| Stokastik | NumPy + SciPy | Generate distribusi Poisson & Eksponensial |
| HTTP Client | httpx | Kirim request dari simulator ke VM |
| Dashboard | Streamlit + Plotly | Visualisasi interaktif real-time |
| Animasi | p5.js v1.9.0 | Animasi packet flow visual |
| Data | pandas | Logging dan analisis hasil simulasi |
| Deploy | Streamlit Cloud + GitHub Pages | Hosting online |

---

## 📄 Lisensi

MIT License.
```