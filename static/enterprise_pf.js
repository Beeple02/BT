(function(){
const _d = v => '$'+v; // dollar-prefix helper
let _pfid = null;
let _pfData = null;
let _charts = {};

// ── Expose to parent ──────────────────────────────────────────────────────────
window.ENT_PF_load = async function(pfid){
  _pfid = pfid;
  await ENT_PF_refresh();
};
window.ENT_PF_refresh = async function(){
  if(!_pfid) return;
  const r = await api(`/api/enterprise/portfolios/${_pfid}/analytics`);
  if(!r.ok){
    const root = document.getElementById('ent-pf-root');
    if(root) root.innerHTML=`<div style="padding:20px;color:var(--dn);font-family:var(--font);font-size:11px">
      Failed to load analytics (HTTP ${r.s}): ${r.d?.detail||'unknown error'}<br>
      <span style="color:var(--txt3)">pfid=${_pfid}</span>
    </div>`;
    return;
  }
  _pfData = r.d;
  renderHeader();
  renderOverview();
  renderPositions();
  renderAnalytics();
  renderRisk();
  renderCashLog();
};

// ── Tab switch ────────────────────────────────────────────────────────────────
window.epfTab = function(tab){
  document.querySelectorAll('.ent-tab').forEach((t,i)=>{
    const tabs=['overview','positions','analytics','risk','report','cashlog'];
    t.classList.toggle('on', tabs[i]===tab);
  });
  document.querySelectorAll('.ent-sub-panel').forEach(p=>{
    p.classList.toggle('on', p.id===`epf-${tab}`);
  });
};

// ── Header strip ──────────────────────────────────────────────────────────────
function renderHeader(){
  const d=_pfData, s=d.summary;
  const set=(id,v,col)=>{const e=document.getElementById(id);if(e){e.textContent=v;if(col)e.style.color=col;}};
  set('epf-client', d.client||'—');
  set('epf-name',   d.name||'—');
  set('epf-value',  '$'+fk(s.total_value));
  set('epf-cash',   '$'+fk(d.cash));
  set('epf-total',  '$'+fk(s.total_with_cash));
  set('epf-pnl',    (s.total_pnl>=0?'+':'')+f(s.total_pnl,2), s.total_pnl>=0?'var(--up)':'var(--dn)');
  set('epf-pnlp',   (s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%', s.total_pnl_pct>=0?'var(--up)':'var(--dn)');
  set('epf-npos',   s.num_positions);
  set('epf-conc',   s.concentration, s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)');
  // Load notes
  const notesEl=document.getElementById('epf-notes-area');
  if(notesEl && d.notes) notesEl.value=d.notes;
}

// ── Overview ──────────────────────────────────────────────────────────────────
function renderOverview(){
  const d=_pfData;
  // Allocation pie
  const ctx=document.getElementById('epf-alloc-chart');
  if(ctx){
    if(_charts.alloc){_charts.alloc.destroy();}
    const labels=d.positions.map(p=>p.ticker);
    if(d.cash>0){labels.push('CASH');}
    const vals=d.positions.map(p=>p.market_value);
    if(d.cash>0) vals.push(d.cash);
    const colors=['#ff8c00','#00e5ff','#00c853','#e040fb','#ffd600','#f44336','#534AB7','#00bfa5','#ff5252','#69f0ae','#555'];
    _charts.alloc=new Chart(ctx,{
      type:'doughnut',
      data:{labels,datasets:[{data:vals,backgroundColor:colors.slice(0,vals.length),borderColor:'#111',borderWidth:2}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'right',labels:{color:'#888',font:{size:9},boxWidth:8,padding:4}},tooltip:{...TT,callbacks:{label:c=>{const total=c.dataset.data.reduce((a,b)=>a+b,0);return ` ${c.label}: ${_d(fk(c.parsed))} (${(c.parsed/total*100).toFixed(1)}%)`;}}}}}
    });
  }
  // Position summary table
  const tbody=document.getElementById('epf-pos-summary');
  if(tbody){
    if(d.positions.length===0){
      tbody.innerHTML='<div style="padding:16px;color:var(--txt3);font-size:10px">No positions. Click + POSITION to add one.</div>';
    } else {
      const posRows = d.positions.map(p=>{
        const pnlCol = p.pnl>=0?'var(--up)':'var(--dn)';
        return '<tr><td style="color:var(--cyn);font-weight:700">'+p.ticker+'</td>'
          +'<td class="r">$'+fk(p.market_value)+'</td>'
          +'<td class="r">'+p.weight_pct.toFixed(1)+'%</td>'
          +'<td class="r" style="color:'+pnlCol+'">'+(p.pnl>=0?'+':'')+'$'+fk(Math.abs(p.pnl))+'</td></tr>';
      }).join('');
      const cashRow = d.cash>0
        ? '<tr style="border-top:1px solid var(--bdr2)"><td style="color:var(--cyn)">CASH</td>'
          +'<td class="r">$'+fk(d.cash)+'</td>'
          +'<td class="r">'+(d.cash/d.summary.total_with_cash*100).toFixed(1)+'%</td>'
          +'<td class="r" style="color:var(--txt3)">—</td></tr>'
        : '';
      tbody.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th class="r">VALUE</th><th class="r">WEIGHT</th><th class="r">P&L</th></tr></thead><tbody>'
        +posRows+cashRow+'</tbody></table>';
    }
  }
  // P&L bars
  const barsEl=document.getElementById('epf-pnl-bars');
  if(barsEl){
    const maxAbs=Math.max(...d.positions.map(p=>Math.abs(p.pnl)),1);
    barsEl.innerHTML=d.positions.map(p=>{
      const w=(Math.abs(p.pnl)/maxAbs*100).toFixed(1);
      return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <span style="width:72px;font-weight:700;color:var(--cyn);font-size:10px">${p.ticker}</span>
        <div style="flex:1;height:8px;background:var(--bdr);border-radius:1px">
          <div style="width:${w}%;height:100%;background:${p.pnl>=0?'var(--up)':'var(--dn)'};border-radius:1px"></div>
        </div>
        <span style="width:70px;text-align:right;font-size:10px;color:${p.pnl>=0?'var(--up)':'var(--dn)'}">${p.pnl>=0?'+':''}${_d(f(p.pnl,2))}</span>
        <span style="width:52px;text-align:right;font-size:9px;color:var(--txt2)">${p.pnl_pct>=0?'+':''}${p.pnl_pct.toFixed(2)}%</span>
      </div>`;
    }).join('');
  }
}

// ── Positions table ───────────────────────────────────────────────────────────
function renderPositions(){
  const tbody=document.getElementById('epf-pos-tbody');
  if(!tbody) return;
  const d=_pfData;
  tbody.innerHTML=d.positions.map(p=>`
    <tr>
      <td style="color:var(--cyn);font-weight:700">${p.ticker}</td>
      <td>${p.qty.toLocaleString()}</td>
      <td class="r">${_d(f(p.entry_price,4))}</td>
      <td class="r" style="color:var(--wht)">${p.live_price!=null?'$'+f(p.live_price,4):'—'}</td>
      <td class="r" style="color:var(--txt2)">${_d(fk(p.cost))}</td>
      <td class="r">${_d(fk(p.market_value))}</td>
      <td class="r" style="color:${p.pnl>=0?'var(--up)':'var(--dn)'}">${p.pnl>=0?'+':''}${_d(f(p.pnl,2))}</td>
      <td class="r" style="color:${p.pnl_pct>=0?'var(--up)':'var(--dn)'}">${p.pnl_pct>=0?'+':''}${p.pnl_pct.toFixed(2)}%</td>
      <td class="r">${p.weight_pct.toFixed(1)}%</td>
      <td class="r" style="color:var(--yel)">${p.ann_vol!=null?p.ann_vol.toFixed(1)+'%':'—'}</td>
      <td class="r" style="color:${(p.sharpe||0)>0?'var(--up)':'var(--dn)'}">${p.sharpe!=null?p.sharpe.toFixed(2):'—'}</td>
      <td style="color:var(--txt2)">${p.entry_date||'—'}</td>
      <td style="color:var(--txt2);max-width:120px;overflow:hidden;text-overflow:ellipsis">${p.notes||''}</td>
      <td><span style="color:var(--dn);cursor:pointer;font-size:11px" onclick="removePosition('${p.id}','${p.ticker}')">✕</span></td>
    </tr>`).join('')||'<tr><td colspan="14" style="color:var(--txt3);padding:16px;text-align:center">No positions</td></tr>';
}

// ── Analytics charts ──────────────────────────────────────────────────────────
function renderAnalytics(){
  const d=_pfData;
  const poss=d.positions.filter(p=>p.ann_vol!=null||p.sharpe!=null);
  // Vol chart
  const vCtx=document.getElementById('epf-vol-chart');
  if(vCtx&&poss.length){
    if(_charts.vol)_charts.vol.destroy();
    _charts.vol=new Chart(vCtx,{
      type:'bar',
      data:{labels:poss.map(p=>p.ticker),datasets:[{data:poss.map(p=>p.ann_vol??0),backgroundColor:poss.map(p=>(p.ann_vol||0)>50?'rgba(244,67,54,.6)':'rgba(255,214,0,.6)'),borderWidth:0}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false},tooltip:{...TT}},scales:{x:{grid:{color:'#1a1a1a'},ticks:{color:'#444'}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',callback:v=>v.toFixed(0)+'%'}}}}
    });
  }
  // Sharpe chart
  const sCtx=document.getElementById('epf-sharpe-chart');
  if(sCtx&&poss.length){
    if(_charts.sharpe)_charts.sharpe.destroy();
    _charts.sharpe=new Chart(sCtx,{
      type:'bar',
      data:{labels:poss.map(p=>p.ticker),datasets:[{data:poss.map(p=>p.sharpe??0),backgroundColor:poss.map(p=>(p.sharpe||0)>0?'rgba(0,200,83,.6)':'rgba(244,67,54,.6)'),borderWidth:0}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false},tooltip:{...TT}},scales:{x:{grid:{color:'#1a1a1a'},ticks:{color:'#444'}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',callback:v=>v.toFixed(2)}}}}
    });
  }
  // Metrics grid
  const grid=document.getElementById('epf-metrics-grid');
  if(grid){
    const s=d.summary;
    const avgVol=d.positions.filter(p=>p.ann_vol).reduce((a,p)=>a+p.ann_vol,0)/Math.max(d.positions.filter(p=>p.ann_vol).length,1);
    const avgSharpe=d.positions.filter(p=>p.sharpe).reduce((a,p)=>a+p.sharpe,0)/Math.max(d.positions.filter(p=>p.sharpe).length,1);
    const mets=[
      {l:'TOTAL RETURN',v:(s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%',c:s.total_pnl_pct>=0?'var(--up)':'var(--dn)'},
      {l:'AVG ANN VOL',v:isNaN(avgVol)?'—':avgVol.toFixed(1)+'%',c:'var(--yel)'},
      {l:'AVG SHARPE',v:isNaN(avgSharpe)?'—':avgSharpe.toFixed(2),c:avgSharpe>0?'var(--up)':'var(--dn)'},
      {l:'HHI',v:s.hhi.toFixed(0),c:s.hhi>3000?'var(--dn)':s.hhi>1500?'var(--yel)':'var(--up)'},
      {l:'NUM POSITIONS',v:s.num_positions,c:'var(--wht)'},
      {l:'CASH',v:'$'+fk(d.cash),c:'var(--cyn)'},
      {l:'INVESTED',v:'$'+fk(s.total_value),c:'var(--wht)'},
      {l:'TOTAL AUM',v:'$'+fk(s.total_with_cash),c:'var(--org)'},
    ];
    grid.innerHTML=mets.map(m=>`<div style="background:var(--bg2);padding:10px 14px;border-right:1px solid var(--bdr)">
      <div style="font-size:9px;color:var(--txt3);letter-spacing:1.5px;margin-bottom:4px">${m.l}</div>
      <div style="font-size:16px;font-weight:700;color:${m.c}">${m.v}</div>
    </div>`).join('');
  }
}

// ── Risk panel ────────────────────────────────────────────────────────────────
function renderRisk(){
  const d=_pfData, s=d.summary;
  // KPI strip
  const kpis=document.getElementById('epf-risk-kpis');
  if(kpis) kpis.innerHTML=`
    <div class="sc"><div class="sc-l">CONCENTRATION</div><div class="sc-v" style="color:${s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)'}">${s.concentration}</div><div class="sc-s">HHI ${s.hhi.toFixed(0)}</div></div>
    <div class="sc"><div class="sc-l">LARGEST POSITION</div><div class="sc-v" style="color:var(--org)">${d.positions.length?d.positions.slice().sort((a,b)=>b.weight_pct-a.weight_pct)[0].ticker:'—'}</div><div class="sc-s">${d.positions.length?d.positions.slice().sort((a,b)=>b.weight_pct-a.weight_pct)[0].weight_pct.toFixed(1)+'%':''}</div></div>
    <div class="sc"><div class="sc-l">WORST P&L</div><div class="sc-v" style="color:var(--dn)">${d.positions.length?d.positions.slice().sort((a,b)=>a.pnl_pct-b.pnl_pct)[0].ticker:'—'}</div><div class="sc-s">${d.positions.length?d.positions.slice().sort((a,b)=>a.pnl_pct-b.pnl_pct)[0].pnl_pct.toFixed(2)+'%':''}</div></div>
    <div class="sc"><div class="sc-l">CASH BUFFER</div><div class="sc-v" style="color:var(--cyn)">${s.total_with_cash>0?(d.cash/s.total_with_cash*100).toFixed(1)+'%':'—'}</div><div class="sc-s">${_d(fk(d.cash))}</div></div>`;
  // Concentration bars
  const concEl=document.getElementById('epf-risk-conc');
  if(concEl) concEl.innerHTML=d.positions.slice().sort((a,b)=>b.weight_pct-a.weight_pct).map(p=>`
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px">
      <span style="width:72px;font-weight:700;font-size:10px;color:var(--cyn)">${p.ticker}</span>
      <div style="flex:1;height:6px;background:var(--bdr)">
        <div style="width:${p.weight_pct.toFixed(1)}%;height:100%;background:${p.weight_pct>30?'var(--dn)':p.weight_pct>15?'var(--yel)':'var(--up)'}"></div>
      </div>
      <span style="width:36px;text-align:right;font-size:9px;color:var(--txt2)">${p.weight_pct.toFixed(1)}%</span>
      ${p.weight_pct>30?'<span style="font-size:7px;color:var(--dn);letter-spacing:1px">HIGH</span>':''}
    </div>`).join('');
  // Kelly
  const kellyEl=document.getElementById('epf-risk-kelly');
  if(kellyEl) kellyEl.innerHTML=d.positions.filter(p=>p.ann_vol).map(p=>{
    const vol=(p.ann_vol||50)/100;
    const ret=(p.pnl_pct||0)/100;
    const kelly=vol>0?Math.max(0,Math.min((ret/(vol*vol))*100,100)).toFixed(1):'—';
    return `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(46,46,46,.4);font-size:10px">
      <span style="color:var(--cyn)">${p.ticker}</span>
      <span style="color:var(--txt2)">Current ${p.weight_pct.toFixed(1)}%</span>
      <span style="color:var(--yel)">Kelly ~${kelly}%</span>
    </div>`;
  }).join('')||'<span style="color:var(--txt3);font-size:10px">Need price history for Kelly calculation.</span>';
  // Flags
  const flagsEl=document.getElementById('epf-risk-flags');
  const flags=[];
  if(s.concentration==='HIGH') flags.push({t:'HIGH CONCENTRATION',d:`HHI ${s.hhi.toFixed(0)} — portfolio is heavily concentrated. Consider diversifying.`,c:'var(--dn)'});
  d.positions.forEach(p=>{ if(p.weight_pct>30) flags.push({t:`OVERWEIGHT: ${p.ticker}`,d:`${p.weight_pct.toFixed(1)}% of portfolio in single position.`,c:'var(--yel)'}); });
  if(d.cash/s.total_with_cash<0.05&&d.positions.length>0) flags.push({t:'LOW CASH BUFFER',d:`Only ${(d.cash/s.total_with_cash*100).toFixed(1)}% cash. Limited dry powder for opportunities.`,c:'var(--yel)'});
  d.positions.forEach(p=>{ if(p.pnl_pct<-20) flags.push({t:`LARGE DRAWDOWN: ${p.ticker}`,d:`${p.pnl_pct.toFixed(2)}% below entry. Review stop-loss.`,c:'var(--dn)'}); });
  if(flagsEl) flagsEl.innerHTML=flags.length===0
    ?'<div style="color:var(--up);font-size:10px">✓ No major risk flags detected.</div>'
    :flags.map(fl=>`<div style="border-left:3px solid ${fl.c};padding:6px 10px;background:rgba(0,0,0,.2);margin-bottom:6px"><div style="font-size:9px;font-weight:700;color:${fl.c};letter-spacing:1px;margin-bottom:2px">${fl.t}</div><div style="font-size:10px;color:var(--txt2)">${fl.d}</div></div>`).join('');
}

// ── Report generator (IBKR-style, sectioned) ─────────────────────────────────
function _rptSec(id){ var el=document.getElementById(id); return el && el.checked; }
function _rptH(title, color){
  return '<div style="font-size:9px;font-weight:700;letter-spacing:2px;color:'+(color||'var(--org)')+';margin:20px 0 8px;border-bottom:1px solid rgba(255,140,0,.2);padding-bottom:4px">'+title+'</div>';
}
function _rptKV(label, value, valColor){
  return '<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(46,46,46,.3)">'
    +'<span style="color:var(--txt2)">'+label+'</span>'
    +'<span style="font-weight:700;color:'+(valColor||'var(--wht)')+'">'+value+'</span></div>';
}
function _rptTable(headers, rows, colAligns){
  var ths = headers.map(function(h,i){
    var align = colAligns&&colAligns[i] ? colAligns[i] : 'left';
    return '<th style="padding:5px 6px;text-align:'+align+';color:var(--txt2);border-bottom:1px solid var(--bdr);white-space:nowrap">'+h+'</th>';
  }).join('');
  var trs = rows.map(function(row){
    var tds = row.map(function(cell, i){
      var align = colAligns&&colAligns[i] ? colAligns[i] : 'left';
      var val = typeof cell === 'object' ? cell.v : cell;
      var col = typeof cell === 'object' ? cell.c : '';
      return '<td style="padding:4px 6px;text-align:'+align+';'+(col?'color:'+col+';':'')+'">'+(val||'—')+'</td>';
    }).join('');
    return '<tr style="border-bottom:1px solid rgba(46,46,46,.25)">'+tds+'</tr>';
  }).join('');
  return '<table style="width:100%;border-collapse:collapse;font-size:10px;margin-bottom:10px">'
    +'<thead><tr>'+ths+'</tr></thead><tbody>'+trs+'</tbody></table>';
}

window.generateReport = function(){
  if(!_pfData) return;
  var d=_pfData, s=d.summary;
  var date=new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'});
  var firm = (document.getElementById('rpt-firm-name')||{}).value || 'BLOOMBERG / NER ENTERPRISE';
  var el=document.getElementById('epf-report-body');
  if(!el) return;

  var html = '<div style="font-family:\'Courier New\',monospace;color:var(--txt);max-width:900px">';

  // ── COVER PAGE ──────────────────────────────────────────────────────────────
  if(_rptSec('rpt-cover')){
    html += '<div style="text-align:center;padding:32px 0 24px;border-bottom:2px solid var(--org);margin-bottom:24px">'
      +'<div style="font-size:9px;color:var(--txt3);letter-spacing:4px;margin-bottom:8px">'+firm.toUpperCase()+'</div>'
      +'<div style="font-size:28px;font-weight:700;color:var(--wht);letter-spacing:2px;margin-bottom:4px">PORTFOLIO STATEMENT</div>'
      +'<div style="font-size:11px;color:var(--txt2);margin-bottom:16px">'+date+'</div>'
      +'<div style="display:inline-grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--bdr);border:1px solid var(--bdr)">'
      +'<div style="background:var(--bg3);padding:10px 20px"><div style="font-size:9px;color:var(--txt3)">CLIENT</div><div style="font-weight:700;color:var(--org)">'+(d.client||'—')+'</div></div>'
      +'<div style="background:var(--bg3);padding:10px 20px"><div style="font-size:9px;color:var(--txt3)">PORTFOLIO</div><div style="font-weight:700;color:var(--wht)">'+(d.name||'—')+'</div></div>'
      +'<div style="background:var(--bg3);padding:10px 20px"><div style="font-size:9px;color:var(--txt3)">TOTAL AUM</div><div style="font-weight:700;color:var(--wht)">$'+fk(s.total_with_cash)+'</div></div>'
      +'</div></div>';
  }

  // ── EXECUTIVE SUMMARY ───────────────────────────────────────────────────────
  if(_rptSec('rpt-summary')){
    var pnlC = s.total_pnl>=0?'var(--up)':'var(--dn)';
    var retC = s.total_pnl_pct>=0?'var(--up)':'var(--dn)';
    var concC = s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)';
    html += _rptH('EXECUTIVE SUMMARY');
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px">';
    html += '<div>';
    html += _rptKV('Report Date', date);
    html += _rptKV('Client', d.client||'—');
    html += _rptKV('Portfolio', d.name||'—');
    if(d.strategy) html += _rptKV('Strategy', d.strategy);
    html += _rptKV('Number of Positions', s.num_positions);
    html += '</div><div>';
    html += _rptKV('Market Value', '$'+fk(s.total_value));
    html += _rptKV('Cash', '$'+fk(d.cash), 'var(--cyn)');
    html += _rptKV('Total AUM', '$'+fk(s.total_with_cash), 'var(--wht)');
    html += _rptKV('Cost Basis', '$'+fk(s.total_cost));
    html += _rptKV('Unrealised P&L', (s.total_pnl>=0?'+':'')+'$'+fk(Math.abs(s.total_pnl))+' ('+s.total_pnl_pct.toFixed(2)+'%)', pnlC);
    html += '</div></div>';

    // Performance KPI boxes
    var avgVol = d.positions.filter(function(p){return p.ann_vol;}).reduce(function(a,p){return a+p.ann_vol;},0) / Math.max(d.positions.filter(function(p){return p.ann_vol;}).length,1);
    var avgShr = d.positions.filter(function(p){return p.sharpe;}).reduce(function(a,p){return a+p.sharpe;},0) / Math.max(d.positions.filter(function(p){return p.sharpe;}).length,1);
    html += '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--bdr);border:1px solid var(--bdr);margin-bottom:12px">';
    var kpis = [
      ['TOTAL RETURN', (s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%', retC],
      ['AVG ANN VOL',  isNaN(avgVol)?'—':avgVol.toFixed(1)+'%', 'var(--yel)'],
      ['AVG SHARPE',   isNaN(avgShr)?'—':avgShr.toFixed(2), avgShr>0?'var(--up)':'var(--dn)'],
      ['HHI',          s.hhi.toFixed(0), concC],
      ['CONCENTRATION',s.concentration, concC]
    ];
    kpis.forEach(function(k){
      html += '<div style="background:var(--bg3);padding:10px;text-align:center">'
        +'<div style="font-size:8px;color:var(--txt3);margin-bottom:4px">'+k[0]+'</div>'
        +'<div style="font-size:15px;font-weight:700;color:'+k[2]+'">'+k[1]+'</div></div>';
    });
    html += '</div>';
  }

  // ── HOLDINGS DETAIL ─────────────────────────────────────────────────────────
  if(_rptSec('rpt-holdings')){
    html += _rptH('HOLDINGS DETAIL');
    var posRows = d.positions.map(function(p){
      var pc=p.pnl>=0?'var(--up)':'var(--dn)';
      var ppc=p.pnl_pct>=0?'var(--up)':'var(--dn)';
      return [
        {v:p.ticker,c:'var(--cyn)'},
        p.qty.toLocaleString(),
        {v:'$'+f(p.entry_price,4)},
        {v:p.live_price!=null?'$'+f(p.live_price,4):'—'},
        {v:'$'+fk(p.market_value)},
        {v:(p.pnl>=0?'+':'')+'$'+f(p.pnl,2), c:pc},
        {v:(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%', c:ppc},
        p.weight_pct.toFixed(1)+'%',
        {v:p.entry_date||'—'},
        p.notes||''
      ];
    });
    html += _rptTable(['TICKER','QTY','ENTRY','LIVE','MKT VALUE','P&L','P&L%','WEIGHT','ENTRY DATE','NOTES'],
      posRows, ['left','right','right','right','right','right','right','right','left','left']);
  }

  // ── PERFORMANCE ANALYSIS ────────────────────────────────────────────────────
  if(_rptSec('rpt-perf')){
    html += _rptH('PERFORMANCE ANALYSIS');
    var sorted_pnl = d.positions.slice().sort(function(a,b){return b.pnl_pct-a.pnl_pct;});
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">';
    html += '<div>'+_rptH('TOP PERFORMERS','var(--up)');
    sorted_pnl.slice(0,3).forEach(function(p){
      var w = Math.min(Math.abs(p.pnl_pct),100).toFixed(0);
      html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        +'<span style="width:52px;color:var(--cyn);font-weight:700">'+p.ticker+'</span>'
        +'<div style="flex:1;height:6px;background:var(--bdr)"><div style="width:'+w+'%;height:100%;background:var(--up)"></div></div>'
        +'<span style="color:var(--up);width:60px;text-align:right">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</span>'
        +'</div>';
    });
    html += '</div><div>'+_rptH('UNDERPERFORMERS','var(--dn)');
    sorted_pnl.slice().reverse().slice(0,3).filter(function(p){return p.pnl_pct<0;}).forEach(function(p){
      var w = Math.min(Math.abs(p.pnl_pct),100).toFixed(0);
      html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        +'<span style="width:52px;color:var(--cyn);font-weight:700">'+p.ticker+'</span>'
        +'<div style="flex:1;height:6px;background:var(--bdr)"><div style="width:'+w+'%;height:100%;background:var(--dn)"></div></div>'
        +'<span style="color:var(--dn);width:60px;text-align:right">'+p.pnl_pct.toFixed(2)+'%</span>'
        +'</div>';
    });
    html += '</div></div>';
  }

  // ── RISK METRICS ─────────────────────────────────────────────────────────────
  if(_rptSec('rpt-risk')){
    html += _rptH('RISK METRICS');
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">';
    html += '<div>';
    html += _rptKV('HHI (Concentration Index)', s.hhi.toFixed(0)+' / 10000', s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)');
    html += _rptKV('Concentration Level', s.concentration, s.concentration==='HIGH'?'var(--dn)':'var(--up)');
    html += _rptKV('Cash Buffer', s.cash_pct ? s.cash_pct.toFixed(1)+'%' : '$'+fk(d.cash));
    var bigPos = d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;})[0];
    if(bigPos) html += _rptKV('Largest Position', bigPos.ticker+' ('+bigPos.weight_pct.toFixed(1)+'%)');
    var worstPos = d.positions.slice().sort(function(a,b){return a.pnl_pct-b.pnl_pct;})[0];
    if(worstPos) html += _rptKV('Worst P&L', worstPos.ticker+' ('+worstPos.pnl_pct.toFixed(2)+'%)', 'var(--dn)');
    html += '</div><div>';
    var withVol = d.positions.filter(function(p){return p.ann_vol!=null;});
    var withShr = d.positions.filter(function(p){return p.sharpe!=null;});
    if(withVol.length){
      var avgV = withVol.reduce(function(a,p){return a+p.ann_vol;},0)/withVol.length;
      var maxV = Math.max.apply(null,withVol.map(function(p){return p.ann_vol;}));
      html += _rptKV('Avg Annualised Volatility', avgV.toFixed(1)+'%', 'var(--yel)');
      html += _rptKV('Highest Vol Position', withVol.find(function(p){return p.ann_vol===maxV;}).ticker+' ('+maxV.toFixed(1)+'%)', 'var(--yel)');
    }
    if(withShr.length){
      var avgS = withShr.reduce(function(a,p){return a+p.sharpe;},0)/withShr.length;
      html += _rptKV('Avg Sharpe Ratio', avgS.toFixed(2), avgS>0?'var(--up)':'var(--dn)');
    }
    html += '</div></div>';
    // Risk flags table
    var flags = [];
    if(s.concentration==='HIGH') flags.push({sev:'HIGH',title:'Concentrated Portfolio',detail:'HHI '+s.hhi.toFixed(0)+' — consider diversifying across more positions.'});
    d.positions.forEach(function(p){
      if(p.weight_pct>30) flags.push({sev:'HIGH',title:'Overweight: '+p.ticker,detail:p.weight_pct.toFixed(1)+'% of portfolio in a single position.'});
      if(p.pnl_pct<-20) flags.push({sev:'HIGH',title:'Large Drawdown: '+p.ticker,detail:p.pnl_pct.toFixed(2)+'% below entry cost. Review stop-loss.'});
    });
    if(s.cash_pct<5 && d.positions.length>0) flags.push({sev:'MED',title:'Low Cash Buffer',detail:'Cash is below 5% of AUM. Limited flexibility for new opportunities.'});
    if(flags.length){
      html += _rptH('RISK FLAGS','var(--dn)');
      html += _rptTable(['SEV','FLAG','DETAIL'], flags.map(function(fl){
        return [{v:fl.sev,c:fl.sev==='HIGH'?'var(--dn)':'var(--yel)'}, {v:fl.title,c:'var(--wht)'}, fl.detail];
      }), ['left','left','left']);
    }
  }

  // ── ALLOCATION BREAKDOWN ─────────────────────────────────────────────────────
  if(_rptSec('rpt-allocation')){
    html += _rptH('ALLOCATION BREAKDOWN');
    var allRows = d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;}).map(function(p){
      var barW = p.weight_pct.toFixed(1);
      var bar = '<div style="display:flex;align-items:center;gap:6px">'
        +'<div style="width:120px;height:5px;background:var(--bdr)"><div style="width:'+barW+'%;height:100%;background:var(--org)"></div></div>'
        +'<span>'+barW+'%</span></div>';
      return [{v:p.ticker,c:'var(--cyn)'}, '$'+fk(p.market_value), bar, '$'+fk(p.cost), (p.pnl>=0?'+':'')+'$'+f(p.pnl,2)];
    });
    // Add cash row
    if(d.cash>0){
      var cashPct = s.total_with_cash>0?(d.cash/s.total_with_cash*100).toFixed(1):'0';
      var cBar = '<div style="display:flex;align-items:center;gap:6px"><div style="width:120px;height:5px;background:var(--bdr)"><div style="width:'+cashPct+'%;height:100%;background:var(--cyn)"></div></div><span>'+cashPct+'%</span></div>';
      allRows.push([{v:'CASH',c:'var(--cyn)'},'$'+fk(d.cash),cBar,'—','—']);
    }
    html += _rptTable(['POSITION','MARKET VALUE','ALLOCATION','COST BASIS','UNREALISED P&L'],
      allRows, ['left','right','left','right','right']);
  }

  // ── PER-POSITION ANALYTICS ───────────────────────────────────────────────────
  if(_rptSec('rpt-individual')){
    html += _rptH('PER-POSITION ANALYTICS');
    var analRows = d.positions.map(function(p){
      return [
        {v:p.ticker,c:'var(--cyn)'},
        {v:p.ann_vol!=null?p.ann_vol.toFixed(1)+'%':'—', c:p.ann_vol>80?'var(--dn)':p.ann_vol>40?'var(--yel)':'var(--up)'},
        {v:p.sharpe!=null?p.sharpe.toFixed(2):'—', c:p.sharpe>1?'var(--up)':p.sharpe>0?'var(--yel)':'var(--dn)'},
        {v:p.sortino!=null?p.sortino.toFixed(2):'—'},
        {v:p.max_dd!=null?p.max_dd.toFixed(2)+'%':'—', c:'var(--dn)'},
        p.weight_pct.toFixed(1)+'%'
      ];
    });
    html += _rptTable(['TICKER','ANN VOL','SHARPE','SORTINO','MAX DD','WEIGHT'],
      analRows, ['left','right','right','right','right','right']);
    html += '<div style="font-size:9px;color:var(--txt3);margin-top:4px">Metrics computed from 30–90 day price history. — indicates insufficient data.</div>';
  }

  // ── CASH ACTIVITY LOG ────────────────────────────────────────────────────────
  if(_rptSec('rpt-cashlog') && d.cash_log && d.cash_log.length){
    html += _rptH('CASH ACTIVITY LOG');
    var cashRows = d.cash_log.slice().reverse().map(function(e){
      var amt = e.amount>=0?'+$'+f(e.amount,2):'-$'+f(Math.abs(e.amount),2);
      return [
        new Date(e.ts).toLocaleDateString('en-GB'),
        {v:amt, c:e.amount>=0?'var(--up)':'var(--dn)'},
        '$'+f(e.balance_after||0,2),
        e.note||'—'
      ];
    });
    html += _rptTable(['DATE','AMOUNT','BALANCE AFTER','NOTE'],
      cashRows, ['left','right','right','left']);
  }

  // ── PORTFOLIO NOTES ──────────────────────────────────────────────────────────
  if(_rptSec('rpt-notes') && d.notes){
    html += _rptH('PORTFOLIO NOTES');
    html += '<div style="background:var(--bg3);padding:12px;color:var(--txt2);line-height:1.7">'+d.notes+'</div>';
  }

  // ── RISK FLAGS ───────────────────────────────────────────────────────────────
  if(_rptSec('rpt-flags')){
    // Already included in risk section above — skip duplicate
  }

  // ── DISCLAIMER ───────────────────────────────────────────────────────────────
  if(_rptSec('rpt-disclaimer')){
    html += '<div style="margin-top:32px;padding-top:16px;border-top:1px solid var(--bdr);font-size:9px;color:var(--txt3);line-height:1.6">'
      +'<div style="font-weight:700;margin-bottom:4px;letter-spacing:1px">DISCLAIMER</div>'
      +'This report is provided for informational purposes only and does not constitute investment advice, a solicitation, or an offer to buy or sell any financial instrument. '
      +'Past performance is not indicative of future results. All figures are based on data available at the time of report generation and may not reflect real-time market conditions. '
      +'The HHI, Sharpe Ratio, and Volatility metrics are computed from historical price data and should be used as one of many tools in investment decision-making. '
      +'</div>';
  }

  html += '<div style="margin-top:24px;padding-top:8px;border-top:1px solid var(--bdr);font-size:9px;color:var(--txt3);text-align:right">'
    +'Generated by '+firm+' · '+date+'</div>';
  html += '</div>';

  el.innerHTML = html;
};

window.printReport = function(){
  var content = document.getElementById('epf-report-body');
  if(!content){ alert('Generate the report first.'); return; }
  var w = window.open('','_blank');
  w.document.write('<html><head><title>Portfolio Report</title><style>'
    +'body{background:#111;color:#bbb;font-family:"Courier New",monospace;padding:30px;font-size:11px;line-height:1.7}'
    +'table{width:100%;border-collapse:collapse} th,td{padding:4px 6px}'
    +'@media print{body{background:#fff;color:#000}}'
    +'</style></head><body>'+content.innerHTML+'</body></html>');
  w.document.close(); w.print();
};

window.exportReportCSV = function(){
  if(!_pfData){ alert('No portfolio data.'); return; }
  var d=_pfData;
  var lines = [
    'PORTFOLIO REPORT - '+d.name+' - '+new Date().toLocaleDateString('en-GB'),
    '',
    'CLIENT,'+d.client,
    'PORTFOLIO,'+d.name,
    'MARKET VALUE,$'+d.summary.total_value,
    'CASH,$'+d.cash,
    'TOTAL AUM,$'+d.summary.total_with_cash,
    'UNREALISED PNL,$'+d.summary.total_pnl,
    'RETURN,%'+d.summary.total_pnl_pct,
    'HHI,'+d.summary.hhi,
    '',
    'HOLDINGS',
    'TICKER,QTY,ENTRY PRICE,LIVE PRICE,MARKET VALUE,PNL,PNL%,WEIGHT%,ANN VOL%,SHARPE,ENTRY DATE'
  ];
  d.positions.forEach(function(p){
    lines.push([
      p.ticker, p.qty, p.entry_price,
      p.live_price!=null?p.live_price:'',
      p.market_value, p.pnl, p.pnl_pct, p.weight_pct,
      p.ann_vol!=null?p.ann_vol:'', p.sharpe!=null?p.sharpe:'',
      p.entry_date||''
    ].join(','));
  });
  if(d.cash_log && d.cash_log.length){
    lines.push('','CASH LOG','DATE,AMOUNT,BALANCE AFTER,NOTE');
    d.cash_log.forEach(function(e){
      lines.push([new Date(e.ts).toLocaleDateString('en-GB'),e.amount,e.balance_after||0,(e.note||'').replace(/,/g,';')].join(','));
    });
  }
  var csv = lines.join('\n');
  var blob = new Blob([csv], {type:'text/csv'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = (d.name||'portfolio').replace(/\s+/g,'_')+'_report.csv'; a.click();
};

// ── Cash log ──────────────────────────────────────────────────────────────────
function renderCashLog(){
  const tbody=document.getElementById('epf-cash-tbody');
  if(!tbody) return;
  const log=(_pfData.cash_log||[]).slice().reverse();
  let running=0;
  const rows=log.map(entry=>{running+=entry.amount;return{...entry,running};});
  tbody.innerHTML=rows.map(r=>`<tr>
    <td style="color:var(--txt2)">${r.ts?new Date(r.ts).toLocaleString('en-GB'):'—'}</td>
    <td style="color:${r.amount>=0?'var(--up)':'var(--dn)'};font-weight:700">${r.amount>=0?'+':''}${f(r.amount,2)}</td>
    <td>${r.note||'—'}</td>
    <td class="r" style="color:var(--wht)">${_d(f(r.running,2))}</td>
  </tr>`).join('')||'<tr><td colspan="4" style="padding:16px;color:var(--txt3);text-align:center">No cash transactions yet.</td></tr>';
}

// ── Notes save ────────────────────────────────────────────────────────────────
window.saveNotes = async function(){
  if(!_pfid||!_pfData) return;
  const notes=document.getElementById('epf-notes-area')?.value||'';
  _pfData.notes=notes;
  const pf=ENT.portfolios.find(p=>p.id===_pfid)||{};
  await apiPost('/api/enterprise/portfolios',{...pf,id:_pfid,notes});
  const bar=document.getElementById('cmd-st');if(bar){bar.textContent='Notes saved';setTimeout(()=>bar.textContent='ENTERPRISE SPACE',1500);}
};

// ── Add position ──────────────────────────────────────────────────────────────
window.showAddPosModal = function(){
  const m=document.getElementById('add-pos-modal');
  if(!m) return;
  // Move to body so position:fixed works correctly
  if(m.parentElement !== document.body) document.body.appendChild(m);
  m.style.display='flex';
};
window.hideAddPosModal = function(){
  const m=document.getElementById('add-pos-modal');
  if(m) m.style.display='none';
};



// ── Cash modal ────────────────────────────────────────────────────────────────
window.showCashModal = function(sign){
  const m=document.getElementById('cash-modal');
  if(!m) return;
  if(m.parentElement !== document.body) document.body.appendChild(m);
  const inp=document.getElementById('cm-amount');
  if(sign&&inp) inp.value=sign>0?'':'-';
  m.style.display='flex';
};
window.hideCashModal = function(){
  const m=document.getElementById('cash-modal');
  if(m) m.style.display='none';
};

window.adjustCash = async function(){
  const amount=parseFloat(document.getElementById('cm-amount').value);
  const note=document.getElementById('cm-note').value;
  if(isNaN(amount)){ alert('Enter a valid amount'); return; }
  const r=await apiPost(`/api/enterprise/portfolios/${_pfid}/cash`,{amount,note});
  if(!r.ok){ alert('Error updating cash'); return; }
  hideCashModal();
  document.getElementById('cm-amount').value='';
  document.getElementById('cm-note').value='';
  await ENT_PF_refresh();
  epfTab('cashlog');
};

// ── Close position mode ───────────────────────────────────────────────────────
window.toggleCloseMode = function(val){
  const isClose = val === 'close';
  const closeFields = document.getElementById('close-pos-fields');
  const addBtn = document.getElementById('add-pos-btn');
  const normalFields = ['apm-ticker','apm-qty','apm-price','apm-date'];
  if(closeFields) closeFields.style.display = isClose ? 'block' : 'none';
  if(addBtn) addBtn.textContent = isClose ? 'CLOSE POSITION \u2192' : 'ADD POSITION \u2192';
  normalFields.forEach(function(id){
    const el = document.getElementById(id);
    if(el){ const row = el.closest('.prow'); if(row) row.style.display = isClose ? 'none' : 'flex'; }
  });
  if(isClose){
    const sel = document.getElementById('close-pos-select');
    if(sel && _pfData){
      sel.innerHTML = '<option value="">— select —</option>'
        + _pfData.positions.map(function(p){
            return '<option value="'+p.id+'">'+p.ticker+' — '+p.qty.toLocaleString()+' shares @ $'+f(p.entry_price,4)+'</option>';
          }).join('');
    }
    const dateEl = document.getElementById('close-date');
    if(dateEl && !dateEl.value) dateEl.value = new Date().toISOString().slice(0,10);
  }
};

window.prefillCloseFields = function(posId){
  if(!posId || !_pfData) return;
  const pos = _pfData.positions.find(function(p){ return p.id === posId; });
  if(!pos) return;
  const qtyEl = document.getElementById('close-qty');
  if(qtyEl) qtyEl.value = pos.qty;
  const priceEl = document.getElementById('close-price');
  if(priceEl) priceEl.value = pos.live_price != null ? pos.live_price : pos.entry_price;
  updateClosePnlPreview();
};

window.updateClosePnlPreview = function(){
  const sel   = document.getElementById('close-pos-select');
  const price = parseFloat(document.getElementById('close-price').value);
  const qty   = parseFloat(document.getElementById('close-qty').value);
  const el    = document.getElementById('close-pnl-val');
  if(!el || !sel || !_pfData) return;
  const pos = _pfData.positions.find(function(p){ return p.id === sel.value; });
  if(!pos || isNaN(price) || isNaN(qty)){ el.textContent = '—'; el.style.color = 'var(--txt2)'; return; }
  const pnl = (price - pos.entry_price) * qty;
  el.textContent = (pnl >= 0 ? '+' : '') + '$' + f(pnl, 2);
  el.style.color = pnl >= 0 ? 'var(--up)' : 'var(--dn)';
};

window.addPosition = async function(){
  const type = document.getElementById('apm-type').value;
  if(type === 'close'){
    const posId      = document.getElementById('close-pos-select').value;
    const closePrice = parseFloat(document.getElementById('close-price').value);
    const closeQty   = parseFloat(document.getElementById('close-qty').value)||null;
    const closeDate  = document.getElementById('close-date').value;
    const notes      = document.getElementById('apm-notes').value;
    if(!posId){ alert('Select a position to close.'); return; }
    if(!closePrice || closePrice <= 0){ alert('Enter a valid close price.'); return; }
    const pos = _pfData && _pfData.positions.find(function(p){ return p.id === posId; });
    const ticker = pos ? pos.ticker : '?';
    const body = {close_price: closePrice, close_date: closeDate, notes: notes};
    if(closeQty) body.close_qty = closeQty;
    const r = await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId+'/close', body);
    if(!r.ok){ alert('Error closing position: '+(r.d.detail||'unknown')); return; }
    hideAddPosModal();
    document.getElementById('apm-type').value = 'long';
    toggleCloseMode('long');
    document.getElementById('apm-notes').value = '';
    await ENT_PF_refresh();
    epfTab('positions');
    const pnl = r.d.realised_pnl;
    const bar = document.getElementById('cmd-st');
    if(bar){
      bar.textContent = (r.d.removed ? 'CLOSED ' : 'PARTIAL CLOSE ') + ticker + ' | PnL: ' + (pnl>=0?'+':'') + '$' + f(pnl,2);
      bar.style.color = pnl>=0 ? 'var(--up)' : 'var(--dn)';
      setTimeout(function(){ bar.textContent='ENTERPRISE SPACE'; bar.style.color='var(--txt3)'; }, 4000);
    }
    return;
  }
  const ticker=(document.getElementById('apm-ticker').value||'').trim().toUpperCase();
  const qty=parseFloat(document.getElementById('apm-qty').value)||0;
  const price=parseFloat(document.getElementById('apm-price').value)||0;
  const date=document.getElementById('apm-date').value;
  const notes=document.getElementById('apm-notes').value;
  if(!ticker||!qty||!price){ alert('Ticker, quantity and entry price are required.'); return; }
  const r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions',{ticker:ticker,qty:qty,entry_price:price,entry_date:date,notes:notes,type:type});
  if(!r.ok){ alert('Error adding position'); return; }
  hideAddPosModal();
  ['apm-ticker','apm-qty','apm-price','apm-notes'].forEach(function(id){ document.getElementById(id).value=''; });
  await ENT_PF_refresh();
  epfTab('positions');
};

window.removePosition = async function(posId, ticker){
  if(!confirm('Remove '+ticker+' from portfolio?')) return;
  await api('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId,{method:'DELETE'});
  await ENT_PF_refresh();
};

// ── Import portfolio ──────────────────────────────────────────────────────────
var _importParsed = [];

window.showImportModal = function(){
  const m = document.getElementById('import-modal');
  if(!m) return;
  if(m.parentElement !== document.body) document.body.appendChild(m);
  document.getElementById('import-raw').value = '';
  document.getElementById('import-preview').style.display = 'none';
  document.getElementById('import-status').style.display = 'none';
  const btn = document.getElementById('import-confirm-btn');
  if(btn) btn.style.display = 'none';
  _importParsed = [];
  m.style.display = 'flex';
};

window.hideImportModal = function(){
  const m = document.getElementById('import-modal');
  if(m) m.style.display = 'none';
};

window.previewImport = function(){
  const raw = (document.getElementById('import-raw').value || '').trim();
  if(!raw){ alert('Paste the NER portfolio table first.'); return; }
  _importParsed = [];
  const lines = raw.split('\n');
  for(var i=0; i<lines.length; i++){
    const trimmed = lines[i].trim();
    if(!trimmed.startsWith('|')) continue;
    const parts = trimmed.split('|').map(function(p){ return p.trim(); }).filter(function(p){ return p.length>0; });
    if(parts.length < 3) continue;
    const ticker = parts[0].toUpperCase();
    if(ticker === 'TICKER' || ticker === '---' || ticker === '' || ticker.startsWith('-') || ticker.startsWith('+')) continue;
    const qty = parseFloat(parts[1].replace(/,/g,''));
    const avgCost = parseFloat(parts[2].replace(/[$,]/g,'').trim());
    if(isNaN(qty) || isNaN(avgCost) || qty <= 0 || avgCost <= 0) continue;
    _importParsed.push({ticker: ticker, qty: qty, entry_price: avgCost});
  }
  if(_importParsed.length === 0){
    const st = document.getElementById('import-status');
    st.style.display = 'block'; st.style.color = 'var(--dn)';
    st.textContent = 'Could not parse any positions. Check the format.';
    return;
  }
  const existing = new Set((_pfData ? _pfData.positions : []).map(function(p){ return p.ticker; }));
  const dupes = _importParsed.filter(function(p){ return existing.has(p.ticker); });
  const rowsEl = document.getElementById('import-preview-rows');
  rowsEl.innerHTML = _importParsed.map(function(p){
    const isDupe = existing.has(p.ticker);
    const style = isDupe ? 'color:var(--txt3);text-decoration:line-through' : 'color:var(--wht)';
    const tag = isDupe ? ' <span style="color:var(--yel);font-size:8px">SKIP</span>' : '';
    return '<div style="'+style+';padding:2px 0">'+p.ticker.padEnd(8)+'  '+String(p.qty).padStart(6)+'  @ $'+f(p.entry_price,4)+tag+'</div>';
  }).join('');
  document.getElementById('import-preview').style.display = 'block';
  const toImport = _importParsed.filter(function(p){ return !existing.has(p.ticker); });
  const st = document.getElementById('import-status');
  st.style.display = 'block';
  st.style.color = toImport.length > 0 ? 'var(--up)' : 'var(--yel)';
  st.textContent = toImport.length + ' position' + (toImport.length===1?'':'s') + ' to import'
    + (dupes.length > 0 ? ', ' + dupes.length + ' skipped (already exist)' : '');
  const btn = document.getElementById('import-confirm-btn');
  if(btn) btn.style.display = toImport.length > 0 ? 'inline-block' : 'none';
};

window.confirmImport = async function(){
  if(!_importParsed.length){ alert('Nothing to import.'); return; }
  const btn = document.getElementById('import-confirm-btn');
  if(btn){ btn.disabled = true; btn.textContent = 'IMPORTING\u2026'; }
  const r = await apiPost('/api/enterprise/portfolios/'+_pfid+'/import', {
    positions: _importParsed.map(function(p){
      return {ticker:p.ticker, qty:p.qty, entry_price:p.entry_price, entry_date:'', notes:'Imported from NER terminal', type:'long'};
    })
  });
  if(btn){ btn.disabled = false; btn.textContent = 'CONFIRM IMPORT \u2192'; }
  if(!r.ok){ alert('Import failed: '+(r.d.detail||'unknown error')); return; }
  hideImportModal();
  await ENT_PF_refresh();
  epfTab('positions');
  const bar = document.getElementById('cmd-st');
  if(bar){
    bar.textContent = 'Imported ' + r.d.imported + ' position' + (r.d.imported===1?'':'s')
      + (r.d.skipped > 0 ? ' (' + r.d.skipped + ' skipped)' : '');
    bar.style.color = 'var(--up)';
    setTimeout(function(){ bar.textContent='ENTERPRISE SPACE'; bar.style.color='var(--txt3)'; }, 4000);
  }
};
})();
