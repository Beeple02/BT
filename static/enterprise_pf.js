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

// ── Report generator ──────────────────────────────────────────────────────────
window.generateReport = function(){
  if(!_pfData) return;
  const d=_pfData, s=d.summary;
  const date=new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'});
  const el=document.getElementById('epf-report-body');
  if(!el) return;

  // Build positions rows with string concat (no nested template literals)
  const posRows=d.positions.map(function(p){
    const pc=p.pnl>=0?'var(--up)':'var(--dn)';
    const ppc=p.pnl_pct>=0?'var(--up)':'var(--dn)';
    return '<tr style="border-bottom:1px solid rgba(46,46,46,.3)">'
      +'<td style="padding:5px 4px;color:var(--cyn);font-weight:700">'+p.ticker+'</td>'
      +'<td style="padding:5px 4px;text-align:right">'+p.qty.toLocaleString()+'</td>'
      +'<td style="padding:5px 4px;text-align:right">$'+f(p.entry_price,4)+'</td>'
      +'<td style="padding:5px 4px;text-align:right">'+(p.live_price!=null?'$'+f(p.live_price,4):'—')+'</td>'
      +'<td style="padding:5px 4px;text-align:right">$'+fk(p.market_value)+'</td>'
      +'<td style="padding:5px 4px;text-align:right;color:'+pc+'">'+(p.pnl>=0?'+':'')+'$'+f(p.pnl,2)+'</td>'
      +'<td style="padding:5px 4px;text-align:right;color:'+ppc+'">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</td>'
      +'<td style="padding:5px 4px;text-align:right">'+p.weight_pct.toFixed(1)+'%</td>'
      +'</tr>';
  }).join('');

  const pnlCol=s.total_pnl>=0?'var(--up)':'var(--dn)';
  const retCol=s.total_pnl_pct>=0?'var(--up)':'var(--dn)';
  const concCol=s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)';
  const notesHtml=d.notes
    ?'<div style="margin-bottom:20px"><div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:8px">NOTES</div>'
      +'<div style="background:var(--bg3);padding:10px;color:var(--txt2)">'+d.notes+'</div></div>'
    :'';

  el.innerHTML=''
    +'<div style="font-family:\'Courier New\',monospace;color:var(--txt)">'
    +'<div style="border-bottom:2px solid var(--org);padding-bottom:12px;margin-bottom:20px">'
    +'<div style="font-size:18px;font-weight:700;color:var(--wht);letter-spacing:2px">PORTFOLIO REPORT</div>'
    +'<div style="font-size:11px;color:var(--txt2);margin-top:4px">'+date+' · BLOOMBERG / NER ENTERPRISE</div>'
    +'</div>'
    +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">'
    +'<div><div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:6px">CLIENT INFORMATION</div>'
    +'<div><b style="color:var(--org)">Client:</b> '+(d.client||'—')+'</div>'
    +'<div><b style="color:var(--org)">Portfolio:</b> '+(d.name||'—')+'</div>'
    +'<div><b style="color:var(--org)">Report Date:</b> '+date+'</div></div>'
    +'<div><div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:6px">PORTFOLIO SUMMARY</div>'
    +'<div><b style="color:var(--org)">Market Value:</b> $'+fk(s.total_value)+'</div>'
    +'<div><b style="color:var(--org)">Cash:</b> $'+fk(d.cash)+'</div>'
    +'<div><b style="color:var(--org)">Total AUM:</b> <span style="color:var(--wht);font-weight:700">$'+fk(s.total_with_cash)+'</span></div></div>'
    +'</div>'
    +'<div style="margin-bottom:20px">'
    +'<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:8px">PERFORMANCE</div>'
    +'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">'
    +'<div style="background:var(--bg3);padding:10px;text-align:center">'
    +'<div style="font-size:9px;color:var(--txt3)">TOTAL P&L</div>'
    +'<div style="font-size:18px;font-weight:700;color:'+pnlCol+'">'+(s.total_pnl>=0?'+':'')+'$'+fk(Math.abs(s.total_pnl))+'</div></div>'
    +'<div style="background:var(--bg3);padding:10px;text-align:center">'
    +'<div style="font-size:9px;color:var(--txt3)">RETURN</div>'
    +'<div style="font-size:18px;font-weight:700;color:'+retCol+'">'+(s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%</div></div>'
    +'<div style="background:var(--bg3);padding:10px;text-align:center">'
    +'<div style="font-size:9px;color:var(--txt3)">CONCENTRATION</div>'
    +'<div style="font-size:18px;font-weight:700;color:'+concCol+'">'+s.concentration+'</div></div>'
    +'</div></div>'
    +'<div style="margin-bottom:20px">'
    +'<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:8px">HOLDINGS</div>'
    +'<table style="width:100%;border-collapse:collapse;font-size:10px">'
    +'<thead><tr style="border-bottom:1px solid var(--bdr)">'
    +'<th style="text-align:left;padding:6px 4px;color:var(--txt2)">TICKER</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">QTY</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">ENTRY</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">LIVE</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">VALUE</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">P&L</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">P&L%</th>'
    +'<th style="text-align:right;padding:6px 4px;color:var(--txt2)">WEIGHT</th>'
    +'</tr></thead><tbody>'+posRows+'</tbody></table></div>'
    +notesHtml
    +'<div style="border-top:1px solid var(--bdr);padding-top:10px;font-size:9px;color:var(--txt3);text-align:center">'
    +'Generated by BLOOMBERG / NER ENTERPRISE · '+date+' · Confidential'
    +'</div></div>';
};

window.printReport = function(){
  const content=document.getElementById('epf-report-body').innerHTML;
  const w=window.open('','_blank');
  w.document.write(`<html><head><title>Portfolio Report</title><style>body{background:#111;color:#bbb;font-family:'Courier New',monospace;padding:30px}*{box-sizing:border-box}</style></head><body>${content}</body></html>`);
  w.document.close();
  w.print();
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
