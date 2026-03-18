// enterprise_firm.js — Firm Dashboard Space JS
(function(){

var _firmData = null;
var _firmCharts = {};
var _modelEdits = {};

window.FIRM_load = async function(){
  var r = await api('/api/enterprise/firm_analytics');
  if(!r.ok){ document.getElementById('firm-inner').innerHTML='<div style="padding:20px;color:var(--dn);font-size:11px">Failed to load firm analytics.</div>'; return; }
  _firmData = r.d;
  FIRM_renderOverview();
  FIRM_renderExposure();
  FIRM_renderModel();
  FIRM_renderCompliance();
  FIRM_renderRevenue();
};

window.FIRM_tab = function(tab){
  document.querySelectorAll('.firm-tab').forEach(function(t,i){
    t.classList.toggle('on',['overview','exposure','model','compliance','revenue'][i]===tab);
  });
  document.querySelectorAll('.firm-panel').forEach(function(p){
    p.classList.toggle('on', p.id==='firm-'+tab);
  });
  // Resize charts after panel becomes visible
  setTimeout(function(){
    Object.values(_firmCharts).forEach(function(c){ if(c&&c.resize) c.resize(); });
  }, 50);
};

function FIRM_renderOverview(){
  var d = _firmData; var s = d.summary;
  // KPIs
  var kpis = document.getElementById('firm-kpis');
  if(kpis){
    kpis.innerHTML=[
      {l:'TOTAL AUM',    v:'$'+fk(s.grand_aum),        c:'var(--wht)'},
      {l:'TOTAL P&L',    v:(s.grand_pnl>=0?'+':'')+'$'+fk(Math.abs(s.grand_pnl)), c:s.grand_pnl>=0?'var(--up)':'var(--dn)'},
      {l:'RETURN',       v:(s.grand_pnl_pct>=0?'+':'')+s.grand_pnl_pct.toFixed(2)+'%', c:s.grand_pnl_pct>=0?'var(--up)':'var(--dn)'},
      {l:'PORTFOLIOS',   v:s.num_portfolios,             c:'var(--wht)'},
      {l:'CLIENTS',      v:s.num_clients,                c:'var(--cyn)'},
    ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');
  }
  // AUM over time chart
  var ctx = document.getElementById('firm-aum-chart');
  if(ctx && d.aum_timeline.length > 1){
    if(_firmCharts.aum) _firmCharts.aum.destroy();
    var labels = d.aum_timeline.map(function(e){return e.date;});
    var vals   = d.aum_timeline.map(function(e){return e.aum;});
    _firmCharts.aum = new Chart(ctx, {
      type:'line',
      data:{labels:labels, datasets:[{data:vals, borderColor:'#ff8c00', borderWidth:2, pointRadius:0, fill:true, backgroundColor:'rgba(255,140,0,.07)'}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false}, tooltip:{...TT, callbacks:{label:function(c){return ' AUM: $'+fk(c.parsed.y);}}}},
        scales:{x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#555',font:{size:9},maxTicksLimit:8}},
                y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#555',font:{size:9},callback:function(v){return '$'+fk(v);}}}}}
    });
  } else if(ctx){
    ctx.parentElement.innerHTML = '<div style="padding:20px;color:var(--txt3);font-size:10px">Add cash deposits to portfolios to build AUM history.</div>';
  }
  // Portfolio pie
  var pctx = document.getElementById('firm-pf-pie');
  if(pctx && d.portfolios.length){
    if(_firmCharts.pie) _firmCharts.pie.destroy();
    var colors = ['#ff8c00','#00e5ff','#00c853','#e040fb','#ffd600','#f44336','#534AB7','#00bfa5'];
    _firmCharts.pie = new Chart(pctx, {
      type:'doughnut',
      data:{labels:d.portfolios.map(function(p){return p.name;}), datasets:[{data:d.portfolios.map(function(p){return p.aum;}), backgroundColor:colors.slice(0,d.portfolios.length), borderColor:'#111', borderWidth:2}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{position:'right',labels:{color:'#888',font:{size:9},boxWidth:8}},tooltip:{...TT,callbacks:{label:function(c){return ' '+c.label+': $'+fk(c.parsed);}}}}}
    });
  }
  // Portfolio table
  var tbl = document.getElementById('firm-pf-table');
  if(tbl){
    tbl.innerHTML = '<thead><tr><th>CLIENT</th><th>PORTFOLIO</th><th class="r">AUM</th><th class="r">VALUE</th><th class="r">CASH</th><th class="r">P&L</th><th class="r">RETURN</th><th class="r">POSITIONS</th><th>LAST ACTIVITY</th></tr></thead>'
      +'<tbody>'+d.portfolios.map(function(p){
        var pc = p.pnl>=0?'var(--up)':'var(--dn)';
        return '<tr>'
          +'<td style="color:var(--org)">'+p.client+'</td>'
          +'<td style="color:var(--wht);font-weight:600">'+p.name+'</td>'
          +'<td class="r" style="font-weight:700">$'+fk(p.aum)+'</td>'
          +'<td class="r">$'+fk(p.value)+'</td>'
          +'<td class="r" style="color:var(--cyn)">$'+fk(p.cash)+'</td>'
          +'<td class="r" style="color:'+pc+'">'+(p.pnl>=0?'+':'')+'$'+fk(Math.abs(p.pnl))+'</td>'
          +'<td class="r" style="color:'+pc+'">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</td>'
          +'<td class="r">'+p.positions+'</td>'
          +'<td style="color:var(--txt2)">'+p.last_activity+'</td>'
          +'</tr>';
      }).join('')+'</tbody>';
  }
}

function FIRM_renderExposure(){
  var d = _firmData;
  // Bar chart
  var ctx = document.getElementById('firm-exposure-chart');
  if(ctx && d.top_exposures.length){
    if(_firmCharts.exp) _firmCharts.exp.destroy();
    _firmCharts.exp = new Chart(ctx, {
      type:'bar',
      data:{
        labels: d.top_exposures.map(function(e){return e.ticker;}),
        datasets:[{
          data: d.top_exposures.map(function(e){return e.pct_of_aum;}),
          backgroundColor: d.top_exposures.map(function(e){return e.pct_of_aum>15?'rgba(244,67,54,.7)':'rgba(255,140,0,.6)';}),
          borderWidth:0
        }]
      },
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return ' '+c.parsed.y.toFixed(2)+'% of firm AUM';}}},},
        scales:{x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#888'}},
                y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#555',callback:function(v){return v.toFixed(0)+'%';}}}}}
    });
  }
  // Concentration flags
  var flagEl = document.getElementById('firm-conc-flags');
  if(flagEl){
    if(!d.concentration_flags.length){
      flagEl.innerHTML = '<div style="color:var(--up);font-size:10px">✓ No ticker exceeds 15% of firm AUM.</div>';
    } else {
      flagEl.innerHTML = d.concentration_flags.map(function(f){
        return '<div style="border-left:3px solid var(--dn);padding:6px 10px;background:rgba(0,0,0,.2);margin-bottom:6px;font-size:10px">'
          +'<b style="color:var(--dn)">'+f.ticker+'</b> — '+f.pct_of_aum.toFixed(1)+'% of firm AUM ($'+fk(f.value)+')'
          +'</div>';
      }).join('');
    }
  }
  // Full table
  var tblEl = document.getElementById('firm-exposure-table');
  if(tblEl){
    tblEl.innerHTML = '<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th class="r">TOTAL VALUE</th><th class="r">% OF FIRM AUM</th></tr></thead><tbody>'
      +d.top_exposures.map(function(e){
        var c = e.pct_of_aum>15?'var(--dn)':e.pct_of_aum>8?'var(--yel)':'var(--txt2)';
        return '<tr><td style="color:var(--cyn);font-weight:700">'+e.ticker+'</td>'
          +'<td class="r">$'+fk(e.value)+'</td>'
          +'<td class="r" style="color:'+c+'">'+e.pct_of_aum.toFixed(2)+'%</td></tr>';
      }).join('')+'</tbody></table>';
  }
}

function FIRM_renderModel(){
  var d = _firmData;
  var editor = document.getElementById('firm-model-editor');
  if(!editor) return;

  // Load saved model from localStorage (client-side only for now)
  var savedModel = {};
  try { savedModel = JSON.parse(localStorage.getItem('firm_model')||'{}'); } catch(e){}

  // Get all unique tickers across portfolios
  var allTickers = {};
  d.portfolios.forEach(function(pf){
    // We don't have per-position data here, use top_exposures
  });
  var exposureTickers = d.top_exposures.map(function(e){return e.ticker;});

  editor.innerHTML = '<div style="font-size:9px;color:var(--txt3);margin-bottom:10px">Set target weights for each ticker in the model portfolio. Portfolios will be compared against this.</div>'
    +'<table class="dt" style="width:100%;max-width:500px"><thead><tr><th>TICKER</th><th class="r">TARGET WEIGHT %</th></tr></thead><tbody>'
    +exposureTickers.map(function(t){
      return '<tr><td style="color:var(--cyn)">'+t+'</td>'
        +'<td class="r"><input type="number" value="'+(savedModel[t]||0)+'" step="1" min="0" max="100" style="width:70px;background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:2px 4px;outline:none" oninput="_firmModelEdit(\''+t+'\',this.value)"></td></tr>';
    }).join('')+'</tbody></table>';

  // Drift per portfolio
  FIRM_renderDrift(savedModel);
}

window._firmModelEdit = function(ticker, val){
  _modelEdits[ticker] = parseFloat(val)||0;
};

window.FIRM_saveModel = function(){
  var model = {};
  document.querySelectorAll('#firm-model-editor input[type=number]').forEach(function(inp,i){
    var ticker = _firmData.top_exposures[i] && _firmData.top_exposures[i].ticker;
    if(ticker) model[ticker] = parseFloat(inp.value)||0;
  });
  Object.assign(model, _modelEdits);
  try { localStorage.setItem('firm_model', JSON.stringify(model)); } catch(e){}
  FIRM_renderDrift(model);
  var bar=document.getElementById('cmd-st');
  if(bar){bar.textContent='Model portfolio saved';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);}
};

function FIRM_renderDrift(model){
  var drift = document.getElementById('firm-model-drift');
  if(!drift||!_firmData) return;
  if(!Object.keys(model).length){
    drift.innerHTML='<div style="color:var(--txt3);font-size:10px">Save a model to see drift per portfolio.</div>';
    return;
  }
  drift.innerHTML='<div style="font-size:9px;color:var(--txt3);margin-bottom:8px">DEMO — per-portfolio drift requires position data from the analytics endpoint.</div>';
}

function FIRM_renderCompliance(){
  var d = _firmData;
  var el = document.getElementById('firm-compliance-list');
  if(!el) return;
  if(!d.compliance.length){
    el.innerHTML='<div style="color:var(--up);font-size:10px;padding:8px">✓ No compliance issues detected.</div>';
    return;
  }
  var typeColors = {INACTIVE:'var(--yel)', CASH_ONLY:'var(--yel)', CONCENTRATION:'var(--dn)'};
  el.innerHTML = d.compliance.map(function(f){
    var c = typeColors[f.type]||'var(--yel)';
    return '<div class="compliance-flag"><span style="color:'+c+';font-weight:700;font-size:9px;letter-spacing:1px">'+f.type+'</span>'
      +' <span style="color:var(--wht)">'+f.pf+'</span>'
      +' — <span style="color:var(--txt2)">'+f.detail+'</span></div>';
  }).join('');
}

function FIRM_renderRevenue(){
  var d = _firmData; var s = d.summary;
  var kpis = document.getElementById('firm-rev-kpis');
  if(kpis){
    kpis.innerHTML=[
      {l:'TOTAL AUM',v:'$'+fk(s.grand_aum),c:'var(--wht)'},
      {l:'EST. MRR (1% AUM)',v:'$'+fk(s.mrr),c:'var(--up)'},
      {l:'EST. ARR',v:'$'+fk(s.arr),c:'var(--up)'},
      {l:'AVG AUM / PORTFOLIO',v:s.num_portfolios?'$'+fk(s.grand_aum/s.num_portfolios):'—',c:'var(--cyn)'},
    ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');
  }
  var tbl = document.getElementById('firm-rev-table');
  if(tbl){
    tbl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>PORTFOLIO</th><th>CLIENT</th><th class="r">AUM</th><th class="r">EST. MRR</th><th class="r">EST. ARR</th></tr></thead><tbody>'
      +d.portfolios.map(function(p){
        var mrr=Math.round(p.aum*0.01/12); var arr=mrr*12;
        return '<tr><td style="color:var(--wht);font-weight:600">'+p.name+'</td><td style="color:var(--org)">'+p.client+'</td>'
          +'<td class="r">$'+fk(p.aum)+'</td><td class="r" style="color:var(--up)">$'+fk(mrr)+'</td><td class="r" style="color:var(--up)">$'+fk(arr)+'</td></tr>';
      }).join('')+'</tbody></table>';
  }
}

})();
