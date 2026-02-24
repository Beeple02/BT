/* Bloomberg Terminal — Shared JS utilities */

// ── Chart.js defaults ────────────────────────────────────────────────────────
if(typeof Chart !== 'undefined') {
  Chart.defaults.color='#555';
  Chart.defaults.borderColor='#1e1e1e';
  Chart.defaults.font.family="'Roboto Mono','Courier New',monospace";
  Chart.defaults.font.size=10;
}

const TT = {
  backgroundColor:'#111',borderColor:'#3a3a3a',borderWidth:1,
  titleColor:'#ccc',bodyColor:'#888',padding:6,
  callbacks:{}
};

// ── API helpers ───────────────────────────────────────────────────────────────
async function api(path, opts={}) {
  try {
    const r = await fetch(path, opts);
    return { ok:r.ok, s:r.status, d:await r.json() };
  } catch(e) { return { ok:false, s:0, d:{detail:e.message} }; }
}
async function apiPost(path, body) {
  return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}

// ── Formatters ────────────────────────────────────────────────────────────────
const f   = (n,d=2) => n==null?'—':typeof n==='number'?n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):n;
const fp  = n => n==null?'—':(n>=0?'+':'')+n.toFixed(2)+'%';
const fk  = n => n==null?'—':n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':n.toLocaleString();
const gc  = n => n>0?'var(--up)':n<0?'var(--dn)':'var(--txt2)';
const ts  = () => new Date().toLocaleTimeString('en-GB');
const dc  = c => { if(c) c.destroy(); return null; };

// ── Chart factories ───────────────────────────────────────────────────────────
function mkLine(ctx, labels, datasets, extra={}) {
  return new Chart(ctx, {
    type:'line', data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{...TT}},
      scales:{
        x:{grid:{color:'#1a1a1a'},ticks:{maxTicksLimit:7,maxRotation:0,color:'#444'}},
        y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444'},...(extra.y||{})}
      },...(extra.opts||{})}
  });
}
function mkBar(ctx, labels, data, colors, extra={}) {
  const bg = Array.isArray(colors)?colors:(data.map(v=>v>=0?'rgba(0,200,83,.6)':'rgba(244,67,54,.6)'));
  return new Chart(ctx, {
    type:'bar', data:{labels,datasets:[{data,backgroundColor:bg,borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{display:false},tooltip:{...TT}},
      scales:{x:{grid:{color:'#1a1a1a'},ticks:{color:'#444'}},
              y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444'}},...(extra.y||{})}}
  });
}
function mkCandle(ctx, candles) {
  return new Chart(ctx,{type:'bar',data:{labels:candles.map(c=>c.date),datasets:[
    {data:candles.map(c=>[c.low,c.high]),
     backgroundColor:candles.map(c=>c.close>=c.open?'rgba(0,200,83,.2)':'rgba(244,67,54,.2)'),
     borderColor:candles.map(c=>c.close>=c.open?'#00c853':'#f44336'),
     borderWidth:1,borderSkipped:false,barPercentage:.5},
    {data:candles.map(c=>[Math.min(c.open,c.close),Math.max(c.open,c.close)]),
     backgroundColor:candles.map(c=>c.close>=c.open?'rgba(0,200,83,.75)':'rgba(244,67,54,.75)'),
     borderWidth:0,borderSkipped:false,barPercentage:.28,categoryPercentage:1}
  ]},options:{responsive:true,maintainAspectRatio:false,animation:false,
    plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:(c)=>{const k=candles[c.dataIndex];return ` O:${f(k.open)} H:${f(k.high)} L:${f(k.low)} C:${f(k.close)}`}}}},
    scales:{x:{grid:{color:'#1a1a1a'},ticks:{maxTicksLimit:8,maxRotation:0,color:'#444'}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444'}}}}});
}
function mkDoughnut(ctx, labels, data, colors) {
  return new Chart(ctx,{type:'doughnut',data:{labels,datasets:[{data,backgroundColor:colors,borderColor:'#111',borderWidth:2}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{position:'right',labels:{color:'#888',font:{size:10},boxWidth:10,padding:6}},
               tooltip:{...TT,callbacks:{label:c=>` ${f(c.parsed)} (${(c.parsed/(c.dataset.data.reduce((a,b)=>a+b,0))*100).toFixed(1)}%)`}}}}});
}

// ── Sparkline canvas ──────────────────────────────────────────────────────────
function spark(data, color, w=60, h=18) {
  const cv=document.createElement('canvas');
  cv.width=w;cv.height=h;cv.style.display='inline-block';cv.style.verticalAlign='middle';
  const ct=cv.getContext('2d');
  if(!data||data.filter(v=>v!=null).length<2) return cv;
  const clean=data.filter(v=>v!=null);
  const mn=Math.min(...clean),mx=Math.max(...clean),rng=mx-mn||1;
  ct.strokeStyle=color||'#555';ct.lineWidth=1.2;ct.beginPath();
  let drawn=0;
  data.forEach((v,i)=>{
    if(v==null) return;
    const x=i/(data.length-1)*(w-2)+1,y=(1-(v-mn)/rng)*(h-2)+1;
    drawn===0?ct.moveTo(x,y):ct.lineTo(x,y);drawn++;
  });
  ct.stroke();return cv;
}

// ── Orderbook renderer ────────────────────────────────────────────────────────
function renderOB(elId, data, rows=10) {
  const el=document.getElementById(elId);if(!el)return null;
  const asks=[...(data.asks||[])].sort((a,b)=>b.price-a.price).slice(0,rows);
  const bids=[...(data.bids||[])].sort((a,b)=>b.price-a.price).slice(0,rows);
  const maxQ=Math.max(...[...asks,...bids].map(o=>o.quantity),1);
  let h='<div style="font-size:11px;font-variant-numeric:tabular-nums">';
  asks.forEach(a=>{const w=(a.quantity/maxQ*85).toFixed(1);
    h+=`<div style="display:flex;padding:1px 8px;position:relative;align-items:center">
      <div style="position:absolute;top:0;bottom:0;right:0;width:${w}%;background:var(--dn);opacity:.15"></div>
      <span style="width:72px;font-weight:600;color:var(--dn)">${f(a.price,4)}</span>
      <span style="flex:1;text-align:right;color:var(--txt2);font-size:10px">${a.quantity.toLocaleString()}</span>
    </div>`;});
  const sp=data.best_ask&&data.best_bid?(data.best_ask-data.best_bid).toFixed(4):null;
  h+=`<div style="display:flex;align-items:center;justify-content:space-between;padding:3px 8px;background:var(--bg3);border-top:1px solid var(--bdr);border-bottom:1px solid var(--bdr);margin:2px 0">
    <span style="color:var(--wht);font-size:13px;font-weight:700">${data.mid!=null?f(data.mid,4):'—'}</span>
    <span style="color:var(--txt3);font-size:10px">SPR ${sp||'—'}</span></div>`;
  bids.forEach(b=>{const w=(b.quantity/maxQ*85).toFixed(1);
    h+=`<div style="display:flex;padding:1px 8px;position:relative;align-items:center">
      <div style="position:absolute;top:0;bottom:0;left:0;width:${w}%;background:var(--up);opacity:.15"></div>
      <span style="width:72px;font-weight:600;color:var(--up)">${f(b.price,4)}</span>
      <span style="flex:1;text-align:right;color:var(--txt2);font-size:10px">${b.quantity.toLocaleString()}</span>
    </div>`;});
  el.innerHTML=h+'</div>';
  return sp;
}

// ── Missing utility — round4 ──────────────────────────────────────────────────
const round4 = n => Math.round(n * 10000) / 10000;

// ── Populate a <select> with securities list ──────────────────────────────────
function populateSel(id, prependAll) {
  const el = document.getElementById(id);
  if (!el) return;
  const cur = el.value;
  const opts = (BB.secs || []).map(s =>
    `<option value="${s.ticker}"${s.ticker===cur?' selected':''}>${s.ticker} — ${s.full_name}</option>`
  ).join('');
  el.innerHTML = (prependAll
    ? '<option value="*">ALL TICKERS</option>'
    : '<option value="">SELECT…</option>') + opts;
}

// ── Show an error banner inside an element ────────────────────────────────────
function showErr(elId, msg) {
  const el = document.getElementById(elId);
  if (el) el.innerHTML = `<div class="err-banner">⚠ ${msg}</div>`;
}

// ── Sort helper ───────────────────────────────────────────────────────────────
function mkSorter(arr, setState) {
  let col=null, dir=1;
  return function(c) {
    if(col===c) dir*=-1; else{col=c;dir=1;}
    document.querySelectorAll('.dt th').forEach(th=>{th.classList.remove('sort-a','sort-d')});
    const ths=[...document.querySelectorAll('.dt th')];
    const th=ths.find(t=>t.dataset.col===c);
    if(th) th.classList.add(dir>0?'sort-a':'sort-d');
    const sorted=[...arr].sort((a,b)=>{
      const av=a[c]??'',bv=b[c]??'';
      return typeof av==='number'?(av-bv)*dir:String(av).localeCompare(String(bv))*dir;
    });
    setState(sorted);
  };
}

// ── Intensity-scaled color (replaces gc for price/change displays) ────────────
// ±0–0.1% → gray, ±0.1–0.5% → faint, ±0.5–2% → normal, ±2–5% → bright, ±5%+ → full+bold
const gcI = (n, el) => {
  if(n == null) return 'var(--txt3)';
  const a = Math.abs(n);
  if(a < 0.1)  return 'var(--txt3)';        // near-flat → neutral gray
  if(n > 0) {
    if(a < 0.5)  return 'rgba(0,200,83,.45)';
    if(a < 2.0)  return 'var(--up)';
    if(a < 5.0)  return '#00e676';
    return '#69ff8a';                         // explosive move
  } else {
    if(a < 0.5)  return 'rgba(244,67,54,.45)';
    if(a < 2.0)  return 'var(--dn)';
    if(a < 5.0)  return '#ff5252';
    return '#ff8a80';
  }
};
// Also sets font-weight so big moves stand out structurally
const gcIW = n => {
  const a = Math.abs(n||0);
  return a >= 5 ? '900' : a >= 2 ? '700' : '500';
};
