import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Simulasi Antrian Server",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
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
.metric-bad  .metric-value { color: #f44336; }
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
.alert-stable   { background: rgba(46,125,50,0.15);  border: 1px solid #2e7d32; border-radius: 8px; padding: 10px 16px; color: #a5d6a7; font-size: 13px; }
.alert-critical { background: rgba(198,40,40,0.15);  border: 1px solid #c62828; border-radius: 8px; padding: 10px 16px; color: #ef9a9a; font-size: 13px; }
.alert-warn     { background: rgba(230,81,0,0.15);   border: 1px solid #e65100; border-radius: 8px; padding: 10px 16px; color: #ffcc80; font-size: 13px; }
.section-header { font-size: 12px; color: #78909c; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(100,181,246,0.15); }
.ctrl-btn button { width: 100% !important; }
.stress-active {
    background: rgba(230,81,0,.15);
    border: 1px solid #e65100;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    color: #ffcc80;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── session_state defaults ─────────────────────────────────────────────────────
for k, v in {"sim_paused": False, "sim_reset_key": 0, "sim_stress": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Simulation logic ───────────────────────────────────────────────────────────
def select_worker(algorithm, request_id, workers_queue, num_workers):
    if algorithm == "round_robin":      return request_id % num_workers
    elif algorithm == "random":         return np.random.randint(0, num_workers)
    elif algorithm == "least_connection": return int(np.argmin(workers_queue))
    return request_id % num_workers


def run_simulation(lambda_rate, num_workers, algorithm, duration, mu):
    results = []
    workers_queue      = [0]   * num_workers
    workers_busy_until = [0.0] * num_workers
    current_time = 0.0
    request_id   = 0
    base_time    = datetime.now()

    while current_time < duration:
        inter_arrival = np.random.exponential(1.0 / lambda_rate)
        current_time += inter_arrival
        if current_time > duration:
            break
        worker_id    = select_worker(algorithm, request_id, workers_queue, num_workers)
        wait_time    = max(0.0, workers_busy_until[worker_id] - current_time)
        service_time = np.random.exponential(1.0 / mu)
        start_service = current_time + wait_time
        finish_time   = start_service + service_time
        workers_busy_until[worker_id] = finish_time
        workers_queue[worker_id] += 1
        results.append({
            "request_id":   request_id,
            "worker_id":    worker_id,
            "arrive_time":  round(current_time,  4),
            "wait_time":    round(wait_time,      4),
            "service_time": round(service_time,   4),
            "total_time":   round(wait_time + service_time, 4),
            "timestamp":    (base_time + timedelta(seconds=current_time)).isoformat(),
        })
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
        title=dict(text=title, font=dict(color="#90caf9", size=14)),
        xaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        yaxis=dict(gridcolor="#1a2a3a", linecolor="#1a2a3a"),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="#0d1422", bordercolor="#1a2a3a", font=dict(size=12)),
    )
    return fig


def get_animation_html(lambda_rate, num_workers, mu, algorithm, speed=1.0, paused=False):
    algo_js = algorithm
    algo_label = {
        "round_robin": "Round Robin",
        "random": "Random",
        "least_connection": "Least Connection",
    }[algorithm]
    algo_desc = {
        "round_robin":      "Round Robin — request didistribusikan bergiliran ke setiap worker",
        "random":           "Random — worker dipilih secara acak setiap request",
        "least_connection": "Least Connection — selalu ke worker dengan antrian terpendek",
    }[algorithm]
    rho       = lambda_rate / (num_workers * mu)
    rho_color = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_sub   = "✅ stabil" if rho < 0.7 else "⚠️ mendekati kritis" if rho < 1 else "🔴 kritis!"
    init_paused = "true" if paused else "false"

    return f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="UTF-8"/>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.9.0/p5.min.js"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{ width:100%; height:100%; overflow:hidden; background:#0a0e1a;
                  font-family:'Segoe UI',Arial,sans-serif; color:#e0e0e0;
                  display:flex; flex-direction:column; }}

    /* ── Info Bar ── */
    #infoBar {{
      flex-shrink:0; display:flex; align-items:stretch;
      background:#0d1422; border-bottom:1px solid #1a2a3a;
    }}
    .info-item {{
      display:flex; flex-direction:column; justify-content:center;
      gap:3px; padding:10px 20px;
      border-right:1px solid #1a2a3a; flex:1;
    }}
    .info-item:last-child {{ border-right:none; }}
    .info-label {{ font-size:10px; color:#546e7a; text-transform:uppercase; letter-spacing:1.2px; font-weight:500; }}
    .info-value {{ font-size:20px; font-weight:700; color:#64b5f6; font-family:monospace; line-height:1.1; }}
    .info-unit  {{ font-size:12px; color:#37474f; font-weight:400; }}
    .info-sub   {{ font-size:10px; color:#37474f; }}

    /* ── Formula bar ── */
    #formulaBar {{
      flex-shrink:0; background:#080c18; border-bottom:1px solid #111827;
      padding:5px 16px; font-size:11px; color:#546e7a;
      display:flex; align-items:center; gap:10px; font-family:monospace;
    }}
    #formulaBar .rhoVal  {{ color:#a5d6a7; font-weight:700; font-size:12px; }}
    #formulaBar .sep     {{ color:#1e3a5f; }}
    #formulaBar .algoDesc {{ color:#5c8a9e; margin-left:auto;
                            font-family:'Segoe UI',sans-serif; font-size:10px; font-style:italic; }}

    /* ── Canvas ── */
    #main {{ flex:1; position:relative; overflow:hidden; }}
    #canvasWrap {{ width:100%; height:100%; position:absolute; inset:0; }}

    #alertBanner {{
      display:none; position:absolute; top:10px; left:50%; transform:translateX(-50%);
      background:rgba(183,28,28,.92); color:#fff; padding:6px 22px;
      border-radius:20px; font-size:12px; font-weight:700;
      border:1px solid #ef5350; animation:abPulse 1s infinite alternate;
      z-index:10; white-space:nowrap; pointer-events:none;
    }}
    #pauseOverlay {{
      display:none; position:absolute; inset:0;
      background:rgba(10,14,26,.65); backdrop-filter:blur(2px);
      z-index:8; align-items:center; justify-content:center;
      flex-direction:column; gap:8px;
    }}
    #pauseOverlay.visible {{ display:flex; }}
    #pauseOverlay span  {{ font-size:28px; font-weight:700; color:#64b5f6; letter-spacing:4px; opacity:.9; }}
    #pauseOverlay small {{ font-size:11px; color:#546e7a; }}

    @keyframes abPulse {{
      from {{ box-shadow:0 0 6px rgba(239,83,80,.4); }}
      to   {{ box-shadow:0 0 20px rgba(239,83,80,.95); }}
    }}
  </style>
</head>
<body>

<!-- Info Bar -->
<div id="infoBar">
  <div class="info-item">
    <div class="info-label">λ — Arrival Rate</div>
    <div class="info-value">{lambda_rate:.1f} <span class="info-unit">req/s</span></div>
    <div class="info-sub">Distribusi Poisson</div>
  </div>
  <div class="info-item">
    <div class="info-label">μ — Service Rate</div>
    <div class="info-value">{mu:.1f} <span class="info-unit">req/s</span></div>
    <div class="info-sub">Distribusi Eksponensial</div>
  </div>
  <div class="info-item">
    <div class="info-label">Workers (c)</div>
    <div class="info-value">{num_workers} <span class="info-unit">server</span></div>
    <div class="info-sub">Paralel aktif</div>
  </div>
  <div class="info-item">
    <div class="info-label">Speed</div>
    <div class="info-value">{speed:.1f}<span class="info-unit">x</span></div>
    <div class="info-sub">Kecepatan simulasi</div>
  </div>
  <div class="info-item">
    <div class="info-label">Algoritma</div>
    <div class="info-value" style="font-size:15px;color:#ce93d8">{algo_label}</div>
    <div class="info-sub">Load balancing</div>
  </div>
  <div class="info-item" style="flex:0.7">
    <div class="info-label">ρ — Utilisasi</div>
    <div class="info-value" style="color:{rho_color}">{rho:.2f}</div>
    <div class="info-sub">{rho_sub}</div>
  </div>
</div>

<!-- Formula Bar -->
<div id="formulaBar">
  <span>ρ = λ/(c·μ) = {lambda_rate:.1f}/({num_workers}×{mu:.1f}) =</span>
  <span class="rhoVal">{rho:.3f}</span>
  <span class="sep">|</span>
  <span>L = λW</span>
  <span class="sep">|</span>
  <span>W = Wq + 1/μ</span>
  <span class="algoDesc">{algo_desc}</span>
</div>

<!-- Canvas -->
<div id="main">
  <div id="canvasWrap">
    <div id="alertBanner">🔴 SISTEM KRITIS — ρ ≥ 1!</div>
    <div id="pauseOverlay" class="{'visible' if paused else ''}">
      <span>⏸ PAUSED</span>
      <small>Lanjutkan dari sidebar Streamlit</small>
    </div>
  </div>
</div>

<script>
  const lambda={lambda_rate}, numWorkers={num_workers}, mu={mu}, speed={speed};
  const algorithm="{algo_js}";
  let paused={init_paused};

  const ALGO_SHORT={{ round_robin:"RR", random:"RND", least_connection:"LC" }};

  let packets=[], workers=[], trails=[];
  let nextSpawn=0, reqId=0, lineAnimOffset=0;

  function expRandom(rate){{ return -Math.log(Math.random())/rate; }}

  function initWorkers(){{
    workers = Array.from({{length:numWorkers}}, (_,i) => ({{
      id:i, queue:0, busy:false, busyTimer:0, processed:0, pendingQueue:[]
    }}));
  }}

  function selectWorker(){{
    if(algorithm==="round_robin") return reqId % numWorkers;
    if(algorithm==="random") return Math.floor(Math.random()*numWorkers);
    return workers.reduce((a,b) => b.queue < a.queue ? b : a).id;
  }}

  new p5(function(p){{
    let W, H;

    p.setup = function(){{
      const wrap = document.getElementById("canvasWrap");
      W = wrap.offsetWidth || window.innerWidth;
      H = wrap.offsetHeight || window.innerHeight - 90;
      let c = p.createCanvas(W, H);
      c.parent("canvasWrap");
      p.textFont("Segoe UI");
      initWorkers();
      new ResizeObserver(() => {{
        const wr = document.getElementById("canvasWrap");
        let nw=wr.offsetWidth, nh=wr.offsetHeight;
        if(nw>10 && nh>10){{ W=nw; H=nh; p.resizeCanvas(W,H); }}
      }}).observe(document.getElementById("canvasWrap"));
    }};

    p.draw = function(){{
      if(paused) return;
      let dt = (p.deltaTime/1000)*speed;
      lineAnimOffset = (lineAnimOffset+dt*28)%12;
      p.background(10,14,26); drawGrid(p,W,H);

      const lbX=W*.32, lbY=H/2, genX=W*.10, genY=H/2;
      const wX=W*.76, wSpacing=H/(numWorkers+1);
      const rho=lambda/(numWorkers*mu);

      document.getElementById("alertBanner").style.display = rho>=1?"block":"none";
      if(rho>=1){{
        let pulse=(Math.sin(p.millis()/300)+1)/2;
        p.noStroke(); p.fill(180,20,20,pulse*18); p.rect(0,0,W,H);
      }}

      for(let w of workers){{
        if(w.busy){{
          w.busyTimer-=dt;
          if(w.busyTimer<=0){{
            w.busy=false; w.queue=Math.max(0,w.queue-1); w.processed++;
            if(w.pendingQueue.length>0){{
              w.pendingQueue.shift();
              w.busy=true; w.busyTimer=expRandom(mu);
            }}
          }}
        }}
      }}

      nextSpawn-=dt;
      if(nextSpawn<=0){{
        let wid=selectWorker(), wy=wSpacing*(wid+1), arriveT=performance.now();
        packets.push({{x:genX,y:genY,tx:wX,ty:wy,wid,phase:"toLB",alpha:255,id:reqId,size:10,arriveT,waitHeat:0}});
        workers[wid].queue++;
        if(!workers[wid].busy){{
          workers[wid].busy=true; workers[wid].busyTimer=expRandom(mu);
        }} else {{
          workers[wid].pendingQueue.push({{arriveT}});
        }}
        reqId++; nextSpawn=expRandom(lambda);
      }}

      for(let t of trails){{
        t.alpha-=8;
        if(t.alpha>0){{ p.noStroke(); p.fill(100,180,255,t.alpha*.2); p.ellipse(t.x,t.y,t.size*.5); }}
      }}
      trails=trails.filter(t=>t.alpha>0);

      drawAnimLine(p,genX+28,genY,lbX-38,lbY,lineAnimOffset);
      for(let i=0;i<numWorkers;i++)
        drawAnimLine(p,lbX+38,lbY,wX-44,wSpacing*(i+1),lineAnimOffset);

      for(let i=0;i<numWorkers;i++){{
        let w=workers[i], wy=wSpacing*(i+1), qLen=w.pendingQueue.length;
        for(let q=0;q<Math.min(qLen,6);q++){{
          let age=w.pendingQueue[q]?(performance.now()-w.pendingQueue[q].arriveT)/1000:0;
          let heat=Math.min(age/3,1);
          let r=Math.round(241-heat*10),g2=Math.round(196-heat*120);
          p.noStroke(); p.fill(r,g2,60,190); p.ellipse(wX-52-q*11,wy,9);
          p.fill(255,200); p.textSize(6); p.textAlign(p.CENTER,p.CENTER); p.text(q+1,wX-52-q*11,wy);
        }}
        if(qLen>6){{ p.fill(231,76,60,200); p.textSize(8); p.textAlign(p.LEFT,p.CENTER); p.text(`+${{qLen-6}}`,wX-52-6*11,wy); }}
      }}

      for(let pk of packets){{
        pk.waitHeat=Math.min((performance.now()-pk.arriveT)/3000,1);
        let r=Math.round(150+pk.waitHeat*81),gb=Math.round(210-pk.waitHeat*134);
        if(pk.phase==="toLB"){{
          pk.x=p.lerp(pk.x,lbX,.09*speed); pk.y=p.lerp(pk.y,lbY,.09*speed);
          if(p.dist(pk.x,pk.y,lbX,lbY)<8) pk.phase="toWorker";
        }} else if(pk.phase==="toWorker"){{
          pk.x=p.lerp(pk.x,pk.tx,.07*speed); pk.y=p.lerp(pk.y,pk.ty,.07*speed);
          if(p.dist(pk.x,pk.y,pk.tx,pk.ty)<8) pk.phase="arrive";
        }} else if(pk.phase==="arrive"){{
          pk.size=p.lerp(pk.size,20,.2); pk.alpha-=12;
          if(pk.alpha<=0) pk.phase="done";
        }}
        if(pk.phase!=="done"&&pk.alpha>0){{
          trails.push({{x:pk.x,y:pk.y,alpha:pk.alpha*.4,size:pk.size}});
          p.noStroke();
          p.fill(r,gb,gb*.6,pk.alpha*.1); p.ellipse(pk.x,pk.y,pk.size*3);
          p.fill(r,gb,gb*.6,pk.alpha*.22); p.ellipse(pk.x,pk.y,pk.size*2);
          p.fill(r,gb,255-pk.waitHeat*200,pk.alpha); p.ellipse(pk.x,pk.y,pk.size);
          p.fill(255,pk.alpha); p.textSize(7); p.textAlign(p.CENTER,p.CENTER); p.text(pk.id,pk.x,pk.y);
        }}
      }}
      packets=packets.filter(pk=>pk.phase!=="done");

      drawBox(p,genX,genY,56,52,[30,80,160],[64,148,230],"REQUEST\\nGENERATOR",`λ=${{lambda}}/s`);
      drawBox(p,lbX,lbY,76,58,[80,40,160],[150,100,230],"LOAD\\nBALANCER",ALGO_SHORT[algorithm]);
      for(let i=0;i<numWorkers;i++){{
        let w=workers[i],wy=wSpacing*(i+1),col=workerColor(w);
        if(w.busy){{ p.noStroke(); p.fill(col[0],col[1],col[2],22+Math.sin(p.millis()/200)*12); p.rect(wX-58,wy-38,116,76,14); }}
        drawWorkerBox(p,wX,wy,w,col,i);
      }}
    }};

    function drawAnimLine(p,x1,y1,x2,y2,off){{
      let d=p.dist(x1,y1,x2,y2);
      p.stroke(30,70,120,70); p.strokeWeight(1); p.line(x1,y1,x2,y2);
      let steps=Math.ceil(d/12);
      for(let i=0;i<steps;i++){{
        let t1=((i*12+off)%d)/d, t2=((i*12+6+off)%d)/d;
        if(t1<0||t1>1||t2<0||t2>1) continue;
        let ax=p.lerp(x1,x2,t1),ay=p.lerp(y1,y2,t1),bx=p.lerp(x1,x2,t2),by=p.lerp(y1,y2,t2);
        p.stroke(100,180,255,170); p.strokeWeight(1.8); p.line(ax,ay,bx,by);
      }}
    }}
    function drawGrid(p,W,H){{
      p.stroke(18,28,46); p.strokeWeight(.5);
      for(let x=0;x<W;x+=40) p.line(x,0,x,H);
      for(let y=0;y<H;y+=40) p.line(0,y,W,y);
    }}
    function drawBox(p,x,y,w,h,cD,cL,title,sub){{
      p.noStroke(); p.fill(0,0,0,60); p.rect(x-w/2+3,y-h/2+3,w,h,10);
      p.fill(cD[0],cD[1],cD[2]); p.rect(x-w/2,y-h/2,w,h,10);
      p.fill(cL[0],cL[1],cL[2],40); p.rect(x-w/2,y-h/2,w,h/2,10,10,0,0);
      p.stroke(cL[0],cL[1],cL[2],130); p.strokeWeight(1.5); p.noFill(); p.rect(x-w/2,y-h/2,w,h,10);
      p.noStroke(); p.fill(220,235,255); p.textSize(9); p.textAlign(p.CENTER,p.CENTER); p.text(title,x,y-8);
      p.fill(cL[0],cL[1],cL[2]); p.textSize(10); p.textStyle(p.BOLD); p.text(sub,x,y+9); p.textStyle(p.NORMAL);
    }}
    function drawWorkerBox(p,x,y,w,col,idx){{
      let bw=88,bh=60;
      p.noStroke(); p.fill(0,0,0,60); p.rect(x-bw/2+3,y-bh/2+3,bw,bh,10);
      p.fill(col[0]*.35,col[1]*.35,col[2]*.35); p.rect(x-bw/2,y-bh/2,bw,bh,10);
      p.fill(col[0],col[1],col[2],50); p.rect(x-bw/2,y-bh/2,bw,bh/2,10,10,0,0);
      p.stroke(col[0],col[1],col[2],180); p.strokeWeight(1.5); p.noFill(); p.rect(x-bw/2,y-bh/2,bw,bh,10);
      p.noStroke();
      p.fill(w.busy?[231,76,60]:[46,204,113]); p.ellipse(x+bw/2-10,y-bh/2+10,7);
      p.fill(220,235,255); p.textSize(9); p.textAlign(p.CENTER,p.CENTER); p.text(`WORKER ${{idx}}`,x,y-14);
      p.fill(col[0]+80,col[1]+80,col[2]+80); p.textSize(10); p.textStyle(p.BOLD); p.text(`Queue: ${{w.queue}}`,x,y+2); p.textStyle(p.NORMAL);
      p.textSize(8); p.fill(w.busy?[231,120,100]:[100,200,130]); p.text(w.busy?"● BUSY":"● IDLE",x,y+17);
      if(w.busy&&w.busyTimer>0){{
        let prog=Math.max(0,Math.min(1,1-w.busyTimer/(1/mu)));
        p.noStroke(); p.fill(30,40,60); p.rect(x-30,y+26,60,5,3);
        p.fill(col[0],col[1],col[2],200); p.rect(x-30,y+26,60*prog,5,3);
      }}
    }}
    function workerColor(w){{
      if(w.queue===0) return[46,204,113];
      if(w.queue<=2)  return[241,196,15];
      return[231,76,60];
    }}
  }});
</script>
</body>
</html>"""


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel Kontrol")

    # Parameter
    st.markdown('<div class="section-header">Parameter Stokastik</div>', unsafe_allow_html=True)
    lambda_rate = st.slider("λ — Arrival Rate (req/s)", 0.5, 5.0, 2.0, 0.5)
    mu          = st.slider("μ — Service Rate (req/s)", 0.5, 3.0, 1.0, 0.5)
    num_workers = st.slider("Jumlah Worker (c)", 1, 5, 3)
    speed       = st.slider("Speed simulasi", 0.5, 3.0, 1.0, 0.5, format="%.1fx")

    rho = lambda_rate / (num_workers * mu)
    rho_color  = "#4caf50" if rho < 0.7 else "#ff9800" if rho < 1 else "#f44336"
    rho_status = "✅ Stabil" if rho < 0.7 else "⚠️ Mendekati kritis" if rho < 1 else "🔴 Tidak stabil!"
    st.markdown(f"""
    <div class="rho-card">
        ρ = λ/(c·μ) = {lambda_rate:.1f}/({num_workers}×{mu:.1f}) =
        <span style="color:{rho_color};font-weight:700;font-size:16px">{rho:.2f}</span><br>
        <span style="color:{rho_color};font-size:12px">{rho_status}</span>
    </div>
    """, unsafe_allow_html=True)

    # Konfigurasi
    st.markdown('<div class="section-header">Konfigurasi Simulasi</div>', unsafe_allow_html=True)
    algorithm = st.selectbox(
        "Algoritma Load Balancer",
        ["round_robin", "random", "least_connection"],
        format_func=lambda x: {"round_robin":"Round Robin","random":"Random","least_connection":"Least Connection"}[x],
    )
    duration = st.slider("Durasi Simulasi (detik)", 10, 60, 30, 10)

    # Kontrol Animasi
    st.markdown('<div class="section-header">Kontrol Animasi</div>', unsafe_allow_html=True)

    # Baris 1 : Pause | Reset
    # col_p, col_r = st.columns(2)
    # with col_p:
    #     pause_label = "▶ Play" if st.session_state.sim_paused else "⏸ Pause"
    #     if st.button(pause_label, use_container_width=True, key="btn_pause"):
    #         st.session_state.sim_paused = not st.session_state.sim_paused
    #         st.rerun()
    # with col_r:
    #     if st.button("↺ Reset", use_container_width=True, key="btn_reset"):
    #         st.session_state.sim_reset_key += 1
    #         st.session_state.sim_paused   = False
    #         st.session_state.sim_stress   = False
    #         st.rerun()

    if st.button("⏸ Pause" if not st.session_state.sim_paused else "▶ Play",
             use_container_width=True, key="btn_pause"):
      st.session_state.sim_paused = not st.session_state.sim_paused
      st.rerun()

    # Baris 2 : Stress Test (full-width)
    stress_label = "⏹ Stop Stress Test" if st.session_state.sim_stress else "📈 Stress Test"
    if st.button(stress_label, use_container_width=True, key="btn_stress"):
        st.session_state.sim_stress = not st.session_state.sim_stress
        st.rerun()

    if st.session_state.sim_stress:
        lambda_rate = 5.0   # paksa λ max
        st.markdown(
            '<div class="stress-active">📈 Stress aktif — λ dipaksa ke 5.0 req/s</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    run = st.button("▶ Tampilkan Hasil Simulasi", use_container_width=True)

    st.markdown('<div class="section-header">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px;color:#546e7a;line-height:1.7">
    <b style="color:#78909c">M/M/c Queue:</b><br>
    • Arrival: Distribusi Poisson (λ)<br>
    • Service: Distribusi Eksponensial (μ)<br>
    • c: Jumlah worker paralel<br>
    • ρ &lt; 1 → sistem stabil
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────
st.markdown("# Simulasi Stokastik Sistem Antrian Server")
st.markdown("**Pemodelan & Simulasi Stokastik** — Distribusi Poisson & Eksponensial | Model M/M/c Queue")
st.divider()

# ── Key unik agar iframe di-remount saat Reset ──────────────────────────────
anim_key = f"anim_{st.session_state.sim_reset_key}"

def show_animation():
    st.components.v1.html(
        get_animation_html(
            lambda_rate=lambda_rate,
            num_workers=num_workers,
            mu=mu,
            algorithm=algorithm,
            speed=speed,
            paused=st.session_state.sim_paused,
        ),
        height=620,
        scrolling=False,
    )

if not run:
    # ── Halaman awal ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:20px">M/M/c</div>
            <div class="metric-sub">Kendall's Notation</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card {'metric-good' if rho<0.7 else 'metric-warn' if rho<1 else 'metric-bad'}">
            <div class="metric-label">Utilisasi ρ saat ini</div>
            <div class="metric-value">{rho:.2f}</div>
            <div class="metric-sub">{rho_status}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Algoritma tersedia</div>
            <div class="metric-value" style="font-size:20px">3</div>
            <div class="metric-sub">RR · Random · LC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-left:4px solid #64b5f6;border-radius:10px;padding:20px 24px;margin-bottom:16px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Cara Penggunaan</div>
        <div style="font-size:13px;color:#90a4ae;line-height:2.2">
        1. Atur <b style="color:#64b5f6">λ</b> — arrival rate request per detik (Distribusi Poisson)<br>
        2. Atur <b style="color:#64b5f6">μ</b> — service rate per worker (Distribusi Eksponensial)<br>
        3. Atur <b style="color:#64b5f6">c</b> — jumlah worker paralel<br>
        4. Atur <b style="color:#64b5f6">Speed</b> — kecepatan animasi simulasi<br>
        5. Pastikan <b style="color:#4caf50">ρ = λ/(c·μ) &lt; 1</b> agar sistem stabil<br>
        6. Gunakan <b style="color:#64b5f6">⏸ Pause / ↺ Reset / 📈 Stress Test</b> di sidebar untuk mengontrol animasi<br>
        7. Klik <b style="color:#64b5f6">▶ Tampilkan Hasil Simulasi</b> untuk lihat hasil lengkap
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1422;border:1px solid #1a2a3a;border-radius:10px;padding:20px 24px;margin-bottom:16px">
        <div style="font-size:12px;color:#546e7a;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px">Rumus Utama</div>
        <div style="font-family:monospace;font-size:13px;color:#90caf9;line-height:2.2">
        ρ = λ / (c × μ) &nbsp;&nbsp;&nbsp; ← utilisasi sistem (harus &lt; 1)<br>
        L = λ × W &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ← Little's Law<br>
        W = Wq + 1/μ &nbsp;&nbsp;&nbsp;&nbsp; ← total time = wait + service
        </div>
    </div>
    """, unsafe_allow_html=True)

    show_animation()

else:
    # ── Run simulation ────────────────────────────────────────────
    with st.spinner("⏳ Menjalankan simulasi stokastik..."):
        results     = run_simulation(lambda_rate, num_workers, algorithm, duration, mu)
        all_results = run_all_algorithms(lambda_rate, num_workers, duration, mu)

    df        = pd.DataFrame(results)
    tput      = len(df) / duration
    avg_wait  = df["wait_time"].mean()
    avg_svc   = df["service_time"].mean()
    avg_total = df["total_time"].mean()
    L         = tput * avg_total

    # Alert
    if rho >= 1:
        st.markdown(f'<div class="alert-critical">🔴 <b>SISTEM TIDAK STABIL</b> — ρ = {rho:.2f} ≥ 1. Antrian akan terus membesar.</div>', unsafe_allow_html=True)
    elif rho >= 0.8:
        st.markdown(f'<div class="alert-warn">⚠️ <b>Mendekati kritis</b> — ρ = {rho:.2f}. Sistem mulai jenuh.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-stable">✅ <b>Sistem stabil</b> — ρ = {rho:.2f}. Kapasitas mencukupi.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric cards
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    metrics = [
        (c1, "Total Request",      len(df),         "sejak reset",                 ""),
        (c2, "Request Selesai",    len(df),          "100% success",                "good"),
        (c3, "Throughput",         f"{tput:.2f}",    "req/s aktual",                ""),
        (c4, "Avg Service Time",   f"{avg_svc:.3f}s","distribusi Exp(μ)",           ""),
        (c5, "Avg Wait Time",      f"{avg_wait:.3f}s","Wq rata-rata",              "warn" if avg_wait>1 else "good"),
        (c6, "Utilisasi ρ",        f"{rho:.2f}",     rho_status,                   "bad" if rho>=1 else "warn" if rho>=0.8 else "good"),
        (c7, "Little's L",         f"{L:.2f}",       f"λ×W={tput:.2f}×{avg_total:.2f}", ""),
    ]
    for col, label, value, sub, cls in metrics:
        col.markdown(f"""
        <div class="metric-card {'metric-'+cls if cls else ''}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Animasi
    show_animation()

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        df["time_bucket"] = (df["arrive_time"] // 5) * 5
        tput_df = df.groupby("time_bucket").size().reset_index(name="count")
        fig1 = px.area(tput_df, x="time_bucket", y="count",
                       labels={"time_bucket":"Waktu (s)","count":"Jumlah Request"})
        fig1.update_traces(line_color="#64b5f6", fillcolor="rgba(100,181,246,0.15)")
        plotly_dark(fig1, "Request per 5 Detik")
        st.plotly_chart(fig1, use_container_width=True, key="fig1")

    with col_r:
        fig2 = px.histogram(df, x="service_time", nbins=25,
                            labels={"service_time":"Service Time (s)","count":"Frekuensi"})
        fig2.update_traces(marker_color="#64b5f6", marker_opacity=0.7)
        x_range = np.linspace(0, df["service_time"].max(), 200)
        y_exp   = mu * np.exp(-mu*x_range) * len(df) * (df["service_time"].max()/25)
        fig2.add_trace(go.Scatter(x=x_range, y=y_exp, mode="lines",
                                   line=dict(color="#ef9f27",width=2,dash="dot"),
                                   name="Exp(μ) teoritis"))
        plotly_dark(fig2, "Distribusi Service Time + Kurva Teoritis")
        st.plotly_chart(fig2, use_container_width=True, key="fig2")

    # ── Charts row 2 ──────────────────────────────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        worker_df = df.groupby("worker_id").size().reset_index(name="requests")
        colors = ["#4caf50" if q<=worker_df["requests"].mean()*0.8
                  else "#ff9800" if q<=worker_df["requests"].mean()*1.2
                  else "#f44336" for q in worker_df["requests"]]
        fig3 = px.bar(worker_df, x="worker_id", y="requests",
                      labels={"worker_id":"Worker ID","requests":"Jumlah Request"})
        fig3.update_traces(marker_color=colors)
        plotly_dark(fig3, f"Request per Worker ({algorithm.replace('_',' ').title()})")
        st.plotly_chart(fig3, use_container_width=True, key="fig3")

    with col_r2:
        fig4 = px.scatter(df, x="arrive_time", y="wait_time", color="worker_id",
                          labels={"arrive_time":"Waktu Kedatangan (s)","wait_time":"Wait Time (s)"},
                          color_continuous_scale="Blues")
        plotly_dark(fig4, "Wait Time vs Waktu Kedatangan")
        st.plotly_chart(fig4, use_container_width=True, key="fig4")

    st.divider()

    # ── Utilisasi per interval ─────────────────────────────────────
    df["interval"] = (df["arrive_time"] // 5) * 5
    util_df = (df.groupby("interval")
                 .apply(lambda g: min(len(g)/(5*lambda_rate), 1.5))
                 .reset_index(name="utilisasi"))
    fig_util = px.area(util_df, x="interval", y="utilisasi",
                       labels={"interval":"Waktu (s)","utilisasi":"Utilisasi Estimasi"})
    fig_util.update_traces(line_color="#ff9800", fillcolor="rgba(255,152,0,0.1)")
    fig_util.add_hline(y=1.0, line_dash="dot", line_color="#f44336",
                       annotation_text="ρ = 1 (batas kritis)", annotation_position="top right")
    fig_util.add_hline(y=rho, line_dash="dash", line_color="#4caf50",
                       annotation_text=f"ρ teoritis = {rho:.2f}", annotation_position="bottom right")
    plotly_dark(fig_util, "Estimasi Utilisasi per Interval Waktu")
    st.plotly_chart(fig_util, use_container_width=True, key="fig_util")

    st.divider()

    # ── Perbandingan algoritma ─────────────────────────────────────
    st.markdown("### Perbandingan Algoritma Load Balancer")
    comparison = []
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            comparison.append({
                "Algoritma":          algo.replace("_"," ").title(),
                "Total Request":      len(d),
                "Throughput (req/s)": round(len(d)/duration, 3),
                "Avg Service Time (s)":round(d["service_time"].mean(), 4),
                "Avg Wait Time (s)":  round(d["wait_time"].mean(), 4),
                "Avg Total Time (s)": round(d["total_time"].mean(), 4),
                "L = λW":             round((len(d)/duration)*d["total_time"].mean(), 3),
            })
    comp_df = pd.DataFrame(comparison)

    col_tbl, col_bar = st.columns([1.2, 1])
    with col_tbl:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    with col_bar:
        fig5 = px.bar(comp_df, x="Algoritma", y="Throughput (req/s)", color="Algoritma",
                      color_discrete_sequence=["#64b5f6","#4caf50","#ff9800"])
        plotly_dark(fig5, "Throughput per Algoritma")
        st.plotly_chart(fig5, use_container_width=True, key="fig5")

    fig6 = go.Figure()
    algo_colors = {"Round Robin":"#64b5f6","Random":"#4caf50","Least Connection":"#ff9800"}
    for algo, res in all_results.items():
        if res:
            d = pd.DataFrame(res)
            name = algo.replace("_"," ").title()
            fig6.add_trace(go.Box(y=d["wait_time"], name=name,
                                   marker_color=algo_colors.get(name,"#64b5f6"), boxmean=True))
    plotly_dark(fig6, "Distribusi Wait Time per Algoritma")
    st.plotly_chart(fig6, use_container_width=True, key="fig6")

    st.divider()

    # ── Little's Law ───────────────────────────────────────────────
    st.markdown("### Validasi Little's Law")
    st.markdown(f"""
    <div class="littles-card">
        <div style="font-size:12px;color:#546e7a;margin-bottom:8px">L = λ × W</div>
        <div class="eq">L = {tput:.3f} × {avg_total:.3f} = {L:.3f}</div>
        <div style="font-size:12px;color:#78909c;margin-top:8px">
        λ aktual = <b style="color:#64b5f6">{tput:.3f} req/s</b> &nbsp;|&nbsp;
        W = <b style="color:#64b5f6">{avg_total:.3f}s</b> &nbsp;|&nbsp;
        L = <b style="color:#64b5f6">{L:.3f} request</b>
        </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Log ────────────────────────────────────────────────────────
    st.markdown("### Log Hasil Simulasi")
    col_log, col_dl = st.columns([4,1])
    with col_log:
        st.dataframe(
            df[["request_id","worker_id","arrive_time","wait_time","service_time","total_time"]],
            use_container_width=True, hide_index=True,
        )
    with col_dl:
        st.markdown("<br><br>", unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "simulation_results.csv", "text/csv")
        st.success("✅ Simulasi selesai!")