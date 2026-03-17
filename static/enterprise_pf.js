(function(){
var _pfid = null;
var _pfData = null;
var _deepData = null;
var _bmData = null;
var _charts = {};
var _targetsEdited = {};

// ── Core ──────────────────────────────────────────────────────────────────────
window.ENT_PF_load = async function(pfid){
  _pfid = pfid;
  _deepData = null; _bmData = null;
  await ENT_PF_refresh();
};
window.ENT_PF_refresh = async function(){
  if(!_pfid) return;
  const r = await api('/api/enterprise/portfolios/'+_pfid+'/analytics');
  if(!r.ok){
    const root = document.getElementById('ent-pf-root');
    if(root) root.innerHTML='<div style="padding:20px;color:var(--dn);font-size:11px">Failed to load (HTTP '+r.s+'): '+(r.d&&r.d.detail||'unknown')+'</div>';
    return;
  }
  _pfData = r.d;
  renderHeader();
  renderOverview();
  renderPositions();
  renderRisk();
  renderCashLog();
  renderRealized();
  renderAuditLog();
  renderTargetAlloc();
};

window.epfTab = function(tab){
  const tabs=['overview','positions','analytics','risk','benchmark','attribution','stress','realized','target','report','cashlog','audit'];
  document.querySelectorAll('.ent-tab').forEach(function(t,i){ t.classList.toggle('on', tabs[i]===tab); });
  document.querySelectorAll('.ent-sub-panel').forEach(function(p){ p.classList.toggle('on', p.id==='epf-'+tab); });
  if(tab==='benchmark') loadBenchmark();
  if(tab==='attribution'||tab==='stress'||tab==='analytics') loadDeepAnalytics();
};

async function loadDeepAnalytics(){
  if(!_pfid) return;
  var days = (document.getElementById('analytics-days')||{}).value || '90';
  // Show loading state
  var equityChart = document.getElementById('epf-equity-chart');
  var metricsEl = document.getElementById('epf-metrics-grid');
  if(metricsEl) metricsEl.innerHTML='<div style="padding:12px;color:var(--txt3);font-size:10px;grid-column:1/-1">Loading analytics...</div>';
  var r = await api('/api/enterprise/portfolios/'+_pfid+'/deep_analytics?days='+days);
  if(!r.ok){
    if(metricsEl) metricsEl.innerHTML='<div style="padding:12px;color:var(--dn);font-size:10px;grid-column:1/-1">Failed to load analytics: '+(r.d&&r.d.detail||'error')+'</div>';
    return;
  }
  if(r.d&&r.d.detail){
    if(metricsEl) metricsEl.innerHTML='<div style="padding:12px;color:var(--yel);font-size:10px;grid-column:1/-1">'+r.d.detail+' — try a longer time window or check that your positions have price history.</div>';
    return;
  }
  if(!r.d || r.d.detail || !r.d.port_closes){
    if(metricsEl) metricsEl.innerHTML='<div style="padding:12px;color:var(--yel);font-size:10px;grid-column:1/-1">'+(r.d&&r.d.detail||'No price history')+' — try a longer time window.</div>';
    return;
  }
  _deepData = r.d;
  renderEquityCurve();
  renderMonthlyReturns();
  renderRollingSharp();
  renderMetricsGrid();
  renderCorrMatrix();
  renderAttribution();
  renderStressTests();
}

async function loadBenchmark(){
  if(!_pfid) return;
  var bm=(document.getElementById('bm-selector')||{}).value||'B:NCOMP';
  var days=(document.getElementById('bm-days')||{}).value||'90';
  var lbl=document.getElementById('bm-label');
  if(lbl) lbl.textContent='Loading...';
  var r = await api('/api/enterprise/portfolios/'+_pfid+'/benchmark?benchmark='+bm+'&days='+days);
  if(!r.ok){ if(lbl) lbl.textContent='Failed'; return; }
  _bmData = r.d;
  if(lbl) lbl.textContent=r.d.benchmark_label||bm;
  var titleEl=document.getElementById('epf-bm-title');
  if(titleEl) titleEl.textContent='PORTFOLIO vs '+(r.d.benchmark_label||bm);
  renderBenchmark();
}

window.loadDeepAnalytics = loadDeepAnalytics;
window.loadBenchmark = loadBenchmark;
window.renderCashLog = function(){ renderCashLog(); };
window.renderAuditLog = function(){ renderAuditLog(); };

// ── Header ────────────────────────────────────────────────────────────────────
function renderHeader(){
  var d=_pfData, s=d.summary;
  var set=function(id,v,col){var e=document.getElementById(id);if(e){e.textContent=v;if(col)e.style.color=col;}};
  set('epf-client', d.client||'—');
  set('epf-name',   d.name||'—');
  set('epf-value',  '$'+fk(s.total_value));
  set('epf-cash',   '$'+fk(d.cash));
  set('epf-total',  '$'+fk(s.total_with_cash));
  set('epf-rpnl',   '$'+fk(d.realized_pnl||0), (d.realized_pnl||0)>=0?'var(--up)':'var(--dn)');
  set('epf-pnl',    (s.total_pnl>=0?'+':'')+f(s.total_pnl,2), s.total_pnl>=0?'var(--up)':'var(--dn)');
  set('epf-pnlp',   (s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%', s.total_pnl_pct>=0?'var(--up)':'var(--dn)');
  set('epf-npos',   s.num_positions);
  set('epf-conc',   'HHI '+s.hhi.toFixed(0)+' ('+s.concentration+')', s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)');
  var notesEl=document.getElementById('epf-notes-area'); if(notesEl&&d.notes&&!notesEl.value) notesEl.value=d.notes;
  var stratEl=document.getElementById('epf-strategy-input'); if(stratEl&&d.strategy&&!stratEl.value) stratEl.value=d.strategy;
}

// ── Overview ──────────────────────────────────────────────────────────────────
function renderOverview(){
  var d=_pfData;
  // Allocation doughnut
  var ctx=document.getElementById('epf-alloc-chart');
  if(ctx){
    if(_charts.alloc){_charts.alloc.destroy();}
    var labels=d.positions.map(function(p){return p.ticker;});
    if(d.cash>0) labels.push('CASH');
    var vals=d.positions.map(function(p){return p.market_value;});
    if(d.cash>0) vals.push(d.cash);
    var colors=['#ff8c00','#00e5ff','#00c853','#e040fb','#ffd600','#f44336','#534AB7','#00bfa5','#ff5252','#69f0ae','#888'];
    _charts.alloc=new Chart(ctx,{type:'doughnut',data:{labels:labels,datasets:[{data:vals,backgroundColor:colors.slice(0,vals.length),borderColor:'#111',borderWidth:2}]},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'right',labels:{color:'#888',font:{size:9},boxWidth:8,padding:4}},tooltip:{...TT,callbacks:{label:function(c){var total=c.dataset.data.reduce(function(a,b){return a+b;},0);return ' '+c.label+': $'+fk(c.parsed)+' ('+( c.parsed/total*100).toFixed(1)+'%)';}}}}}});
  }
  // Position summary
  var sum=document.getElementById('epf-pos-summary');
  if(sum){
    if(d.positions.length===0){
      sum.innerHTML='<div style="padding:16px;color:var(--txt3);font-size:10px">No positions. Click + POSITION to add one.</div>';
    } else {
      var rows=d.positions.map(function(p){
        var pc=p.pnl>=0?'var(--up)':'var(--dn)';
        return '<tr><td style="color:var(--cyn);font-weight:700">'+p.ticker+'</td>'
          +'<td class="r">$'+fk(p.market_value)+'</td>'
          +'<td class="r">'+p.weight_pct.toFixed(1)+'%</td>'
          +'<td class="r" style="color:'+pc+'">'+(p.pnl>=0?'+':'')+'$'+fk(Math.abs(p.pnl))+'</td></tr>';
      }).join('');
      var cashRow=d.cash>0?'<tr><td style="color:var(--cyn)">CASH</td><td class="r">$'+fk(d.cash)+'</td><td class="r">'+(d.cash/d.summary.total_with_cash*100).toFixed(1)+'%</td><td class="r" style="color:var(--txt3)">—</td></tr>':'';
      sum.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th class="r">VALUE</th><th class="r">WEIGHT</th><th class="r">P&L</th></tr></thead><tbody>'+rows+cashRow+'</tbody></table>';
    }
  }
  // P&L bars
  var barsEl=document.getElementById('epf-pnl-bars');
  if(barsEl){
    var maxAbs=Math.max.apply(null,d.positions.map(function(p){return Math.abs(p.pnl);}));
    if(maxAbs===0) maxAbs=1;
    barsEl.innerHTML=d.positions.map(function(p){
      var w=(Math.abs(p.pnl)/maxAbs*100).toFixed(1);
      return '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        +'<span style="width:60px;font-weight:700;color:var(--cyn);font-size:10px">'+p.ticker+'</span>'
        +'<div style="flex:1;height:8px;background:var(--bdr)"><div style="width:'+w+'%;height:100%;background:'+(p.pnl>=0?'var(--up)':'var(--dn)')+'"></div></div>'
        +'<span style="width:80px;text-align:right;font-size:10px;color:'+(p.pnl>=0?'var(--up)':'var(--dn)')+'">'+(p.pnl>=0?'+':'')+'$'+f(p.pnl,2)+'</span>'
        +'<span style="width:52px;text-align:right;font-size:9px;color:var(--txt2)">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</span>'
        +'</div>';
    }).join('');
  }
}

// ── Positions table ───────────────────────────────────────────────────────────
function renderPositions(){
  var tbody=document.getElementById('epf-pos-tbody');
  if(!tbody||!_pfData) return;
  var d=_pfData;
  tbody.innerHTML=d.positions.map(function(p){
    var pc=p.pnl>=0?'var(--up)':'var(--dn)';
    var ppc=p.pnl_pct>=0?'var(--up)':'var(--dn)';
    var typeBadge=p.type==='short'?'<span style="color:var(--dn);font-size:8px">SHORT</span>':'<span style="color:var(--up);font-size:8px">LONG</span>';
    var tagHtml=p.tag?'<span style="font-size:7px;padding:1px 5px;border:1px solid var(--bdr2);color:var(--yel);margin-left:3px">'+p.tag.toUpperCase()+'</span>':'';
    var stopHtml='—';
    if(p.stop_price&&p.stop_price>0&&p.live_price){
      var dist=(p.live_price-p.stop_price)/p.live_price*100;
      var stopCol=dist<5?'var(--dn)':dist<15?'var(--yel)':'var(--up)';
      stopHtml='<span style="color:'+stopCol+'">'+dist.toFixed(1)+'%</span>';
    }
    return '<tr>'
      +'<td style="color:var(--cyn);font-weight:700">'+p.ticker+tagHtml+'</td>'
      +'<td>'+typeBadge+'</td>'
      +'<td>'+p.qty.toLocaleString()+'</td>'
      +'<td class="r">$'+f(p.entry_price,4)+'</td>'
      +'<td class="r" style="color:var(--wht)">'+(p.live_price!=null?'$'+f(p.live_price,4):'—')+'</td>'
      +'<td class="r" style="color:var(--txt2)">$'+fk(p.cost)+'</td>'
      +'<td class="r">$'+fk(p.market_value)+'</td>'
      +'<td class="r" style="color:'+pc+'">'+(p.pnl>=0?'+':'')+'$'+f(p.pnl,2)+'</td>'
      +'<td class="r" style="color:'+ppc+'">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</td>'
      +'<td class="r">'+p.weight_pct.toFixed(1)+'%</td>'
      +'<td class="r" style="color:var(--yel)">'+(p.ann_vol!=null?p.ann_vol.toFixed(1)+'%':'—')+'</td>'
      +'<td class="r" style="color:'+((p.sharpe||0)>0?'var(--up)':'var(--dn)')+'">'+(p.sharpe!=null?p.sharpe.toFixed(2):'—')+'</td>'
      +'<td class="r">'+(p.sortino!=null?p.sortino.toFixed(2):'—')+'</td>'
      +'<td class="r" style="color:var(--dn)">'+(p.max_dd!=null?p.max_dd.toFixed(2)+'%':'—')+'</td>'
      +'<td class="r">'+stopHtml+'</td>'
      +'<td style="color:var(--txt2)">'+(p.entry_date||'—')+'</td>'
      +'<td style="color:var(--txt2);max-width:80px;overflow:hidden;text-overflow:ellipsis">'+(p.notes||'')+'</td>'
      +'<td>'
      +'<span style="color:var(--txt3);cursor:pointer;font-size:9px;margin-right:4px" onclick="showPositionCurve(&quot;'+p.id+'&quot;,&quot;'+p.ticker+'&quot;)">📈</span>'
      +'<span style="color:var(--dn);cursor:pointer;font-size:11px" onclick="removePosition(&quot;'+p.id+'&quot;,&quot;'+p.ticker+'&quot;)">✕</span>'
      +'</td>'
      +'</tr>';
  }).join('')||'<tr><td colspan="18" style="color:var(--txt3);padding:16px;text-align:center">No positions</td></tr>';
}

// ── Analytics: equity curve, monthly returns, rolling Sharpe, metrics, corr ─
// Shared crosshair plugin for line charts
var _crosshairPlugin = {
  id: 'crosshair',
  afterDraw: function(chart) {
    if(!chart._hoverX) return;
    var ctx2 = chart.ctx;
    var xScale = chart.scales.x;
    var yScale = chart.scales.y;
    ctx2.save();
    ctx2.beginPath();
    ctx2.moveTo(chart._hoverX, yScale.top);
    ctx2.lineTo(chart._hoverX, yScale.bottom);
    ctx2.lineWidth = 1;
    ctx2.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx2.setLineDash([3,3]);
    ctx2.stroke();
    ctx2.restore();
  },
  afterEvent: function(chart, args) {
    var e = args.event;
    if(e.type === 'mousemove') {
      chart._hoverX = e.x;
    } else if(e.type === 'mouseout') {
      chart._hoverX = null;
    }
    chart.draw();
  }
};

function renderEquityCurve(){
  if(!_deepData) return;
  var ctx=document.getElementById('epf-equity-chart');
  if(!ctx) return;
  if(_charts.equity) _charts.equity.destroy();
  var pts=_deepData.port_closes;
  var refDates=_deepData.ref_dates||null;
  var labels=pts.map(function(_,i){return refDates&&refDates[i]?refDates[i]:'D'+i;});
  var up=pts[pts.length-1]>=pts[0];
  var lineCol=up?'#00c853':'#f44336';
  var fillCol=up?'rgba(0,200,83,.07)':'rgba(244,67,54,.06)';

  // Find high/low for annotation
  var maxVal=Math.max.apply(null,pts); var minVal=Math.min.apply(null,pts);
  var maxIdx=pts.indexOf(maxVal);   var minIdx=pts.indexOf(minVal);

  _charts.equity=new Chart(ctx,{
    type:'line',
    data:{labels:labels,datasets:[
      {data:pts,borderColor:lineCol,borderWidth:2,pointRadius:0,
       pointHoverRadius:5,pointHoverBackgroundColor:lineCol,
       fill:true,backgroundColor:fillCol,tension:0.1}
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{display:false},
        tooltip:{
          backgroundColor:'rgba(15,15,15,0.92)',
          borderColor:'rgba(255,140,0,0.4)',
          borderWidth:1,
          titleColor:'#ff8c00',
          bodyColor:'#ccc',
          padding:10,
          callbacks:{
            title:function(items){
              var i=items[0].dataIndex;
              return labels[i]||('Day '+i);
            },
            label:function(c){
              var v=c.parsed.y;
              var ret=(v/10000-1)*100;
              return '  Value: $'+f(v,2)+'  ('+( ret>=0?'+':'')+ret.toFixed(2)+'%)';
            },
            afterLabel:function(c){
              var i=c.dataIndex;
              if(i===0) return '';
              var dayRet=(pts[i]-pts[i-1])/pts[i-1]*100;
              return '  Day: '+(dayRet>=0?'+':'')+dayRet.toFixed(3)+'%';
            }
          }
        }
      },
      scales:{
        x:{
          grid:{color:'rgba(46,46,46,.3)'},
          ticks:{color:'#555',maxTicksLimit:10,maxRotation:0,font:{size:9}}
        },
        y:{
          grid:{color:'rgba(30,30,30,.8)'},
          position:'right',
          ticks:{color:'#555',font:{size:9},callback:function(v){return '$'+fk(v);}}
        }
      }
    },
    plugins:[_crosshairPlugin]
  });

  // Annotate peak and trough directly on canvas after render
  _charts.equity._peakIdx = maxIdx;
  _charts.equity._troughIdx = minIdx;

  // Drag-select to zoom (simple: mousedown+mousemove+mouseup on canvas)
  _initChartZoom(ctx, _charts.equity, pts, labels);
}

// Lightweight drag-to-zoom for a line chart
function _initChartZoom(canvas, chart, pts, labels){
  var drag={active:false, startX:0, startIdx:0};
  var overlay=null;

  canvas.onmousedown=function(e){
    if(!e.shiftKey) return; // only zoom when shift held
    drag.active=true; drag.startX=e.offsetX;
    drag.startIdx=_xToIdx(chart,e.offsetX,pts.length);
    if(!overlay){ overlay=document.createElement('div'); overlay.style.cssText='position:absolute;top:0;bottom:0;background:rgba(255,140,0,.08);border:1px solid rgba(255,140,0,.3);pointer-events:none'; canvas.parentElement.style.position='relative'; canvas.parentElement.appendChild(overlay); }
    overlay.style.display='block'; overlay.style.left=e.offsetX+'px'; overlay.style.width='0';
  };
  canvas.onmousemove=function(e){
    if(!drag.active||!overlay) return;
    var x1=Math.min(drag.startX,e.offsetX);
    var x2=Math.max(drag.startX,e.offsetX);
    overlay.style.left=x1+'px'; overlay.style.width=(x2-x1)+'px';
  };
  canvas.onmouseup=function(e){
    if(!drag.active) return; drag.active=false;
    if(overlay) overlay.style.display='none';
    var endIdx=_xToIdx(chart,e.offsetX,pts.length);
    var i1=Math.min(drag.startIdx,endIdx);
    var i2=Math.max(drag.startIdx,endIdx);
    if(i2-i1<2) return;
    chart.data.labels=labels.slice(i1,i2+1);
    chart.data.datasets[0].data=pts.slice(i1,i2+1);
    chart.update('none');
  };
  canvas.ondblclick=function(){
    // Reset zoom
    chart.data.labels=labels;
    chart.data.datasets[0].data=pts;
    chart.update('none');
  };
}

function _xToIdx(chart,x,total){
  var xScale=chart.scales.x;
  if(!xScale) return 0;
  var pct=(x-xScale.left)/(xScale.right-xScale.left);
  return Math.max(0,Math.min(total-1,Math.round(pct*(total-1))));
}

function renderMonthlyReturns(){
  if(!_deepData||!_deepData.monthly_returns.length) return;
  var el=document.getElementById('epf-monthly-returns');
  if(!el) return;
  var html='<div style="display:flex;flex-wrap:wrap;gap:4px">';
  _deepData.monthly_returns.forEach(function(m){
    var pct=m.return_pct;
    var intensity=Math.min(Math.abs(pct)/5,1);
    var bg=pct>=0?'rgba(0,200,83,'+intensity*0.7+')':'rgba(244,67,54,'+intensity*0.7+')';
    html+='<div style="background:'+bg+';padding:4px 8px;min-width:80px;text-align:center">'
      +'<div style="font-size:8px;color:var(--txt3)">'+m.month+'</div>'
      +'<div style="font-size:10px;font-weight:700;color:var(--wht)">'+(pct>=0?'+':'')+pct.toFixed(2)+'%</div>'
      +'</div>';
  });
  html+='</div>';
  el.innerHTML=html;
}

function renderRollingSharp(){
  if(!_deepData||!_deepData.rolling_sharpe.length) return;
  var ctx=document.getElementById('epf-rolling-sharpe-chart');
  if(!ctx) return;
  if(_charts.rsharp) _charts.rsharp.destroy();
  var pts=_deepData.rolling_sharpe;
  var colors=pts.map(function(v){return v>=0?'rgba(0,200,83,.8)':'rgba(244,67,54,.8)';});
  _charts.rsharp=new Chart(ctx,{
    type:'bar',
    data:{labels:pts.map(function(_,i){return i;}),datasets:[{data:pts,backgroundColor:colors,borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{display:false},tooltip:{...TT}},
      scales:{x:{display:false},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',callback:function(v){return v.toFixed(1);}}}}}});
}

function renderMetricsGrid(){
  var el=document.getElementById('epf-metrics-grid');
  if(!el) return;
  var m=_deepData&&_deepData.portfolio_metrics;
  var s=_pfData&&_pfData.summary;
  var mets=[
    {l:'TWR',v:m?(m.twr>=0?'+':'')+m.twr.toFixed(2)+'%':'—',c:m&&m.twr>=0?'var(--up)':'var(--dn)'},
    {l:'MWR',v:m?(m.mwr>=0?'+':'')+m.mwr.toFixed(2)+'%':'—',c:m&&m.mwr>=0?'var(--up)':'var(--dn)'},
    {l:'SHARPE',v:m?m.sharpe.toFixed(2):'—',c:m&&m.sharpe>0?'var(--up)':'var(--dn)'},
    {l:'SORTINO',v:m?m.sortino.toFixed(2):'—',c:m&&m.sortino>0?'var(--up)':'var(--dn)'},
    {l:'CALMAR',v:m?m.calmar.toFixed(2):'—',c:'var(--yel)'},
    {l:'ANN VOL',v:m?m.ann_vol.toFixed(1)+'%':'—',c:'var(--yel)'},
    {l:'MAX DD',v:m?'-'+m.max_drawdown.toFixed(2)+'%':'—',c:'var(--dn)'},
    {l:'HHI',v:s?s.hhi.toFixed(0):'—',c:s&&s.concentration==='HIGH'?'var(--dn)':'var(--up)'},
    {l:'POSITIONS',v:s?s.num_positions:'—',c:'var(--wht)'},
    {l:'CASH %',v:s?s.cash_pct.toFixed(1)+'%':'—',c:'var(--cyn)'},
  ];
  el.innerHTML=mets.map(function(m){
    return '<div style="background:var(--bg2);padding:10px 14px">'
      +'<div style="font-size:8px;color:var(--txt3);letter-spacing:1.5px;margin-bottom:4px">'+m.l+'</div>'
      +'<div style="font-size:15px;font-weight:700;color:'+m.c+'">'+m.v+'</div>'
      +'</div>';
  }).join('');
}

function renderCorrMatrix(){
  if(!_deepData||!_deepData.tickers.length) return;
  var el=document.getElementById('epf-corr-matrix');
  if(!el) return;
  var tickers=_deepData.tickers;
  var corr=_deepData.correlation;
  var html='<table style="border-collapse:collapse;font-size:9px">';
  html+='<tr><td style="padding:2px 6px"></td>';
  tickers.forEach(function(t){html+='<th style="padding:2px 8px;color:var(--txt2);text-align:center">'+t+'</th>';});
  html+='</tr>';
  tickers.forEach(function(t1){
    html+='<tr><th style="padding:2px 8px;color:var(--cyn);text-align:left">'+t1+'</th>';
    tickers.forEach(function(t2){
      var v=corr[t1]&&corr[t1][t2]!=null?corr[t1][t2]:0;
      var abs=Math.abs(v);
      var bg=t1===t2?'rgba(255,140,0,.3)':v>0?'rgba(0,200,83,'+abs*0.6+')':'rgba(244,67,54,'+abs*0.6+')';
      html+='<td style="background:'+bg+';padding:3px 6px;text-align:center;min-width:40px;font-weight:700;color:var(--wht)">'+v.toFixed(2)+'</td>';
    });
    html+='</tr>';
  });
  html+='</table>';
  el.innerHTML=html;
}

// ── Risk ──────────────────────────────────────────────────────────────────────
function renderRisk(){
  var d=_pfData, s=d.summary;
  // KPI strip
  var kpis=document.getElementById('epf-risk-kpis');
  if(kpis){
    var pm=_deepData&&_deepData.portfolio_metrics;
    var varv=_deepData?_deepData.var95_pct:null;
    var cvarv=_deepData?_deepData.cvar95_pct:null;
    kpis.innerHTML=[
      {l:'CONCENTRATION',v:s.concentration,s:'HHI '+s.hhi.toFixed(0),c:s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)'},
      {l:'LARGEST POSITION',v:d.positions.length?d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;})[0].ticker:'—',s:d.positions.length?d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;})[0].weight_pct.toFixed(1)+'%':'',c:'var(--org)'},
      {l:'WORST P&L',v:d.positions.length?d.positions.slice().sort(function(a,b){return a.pnl_pct-b.pnl_pct;})[0].ticker:'—',s:d.positions.length?d.positions.slice().sort(function(a,b){return a.pnl_pct-b.pnl_pct;})[0].pnl_pct.toFixed(2)+'%':'',c:'var(--dn)'},
      {l:'VAR 95%',v:varv!=null?'-'+varv.toFixed(2)+'%':'—',s:cvarv!=null?'CVaR: -'+cvarv.toFixed(2)+'%':'load Analytics tab',c:'var(--dn)'},
      {l:'CASH BUFFER',v:s.cash_pct.toFixed(1)+'%',s:'$'+fk(d.cash),c:'var(--cyn)'},
    ].map(function(k){
      return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div>'+(k.s?'<div class="sc-s">'+k.s+'</div>':'')+'</div>';
    }).join('');
  }
  // Concentration bars
  var concEl=document.getElementById('epf-risk-conc');
  if(concEl) concEl.innerHTML=d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;}).map(function(p){
    return '<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px">'
      +'<span style="width:60px;font-weight:700;font-size:10px;color:var(--cyn)">'+p.ticker+'</span>'
      +'<div style="flex:1;height:6px;background:var(--bdr)"><div style="width:'+p.weight_pct.toFixed(1)+'%;height:100%;background:'+(p.weight_pct>30?'var(--dn)':p.weight_pct>15?'var(--yel)':'var(--up)')+'"></div></div>'
      +'<span style="width:40px;text-align:right;font-size:9px;color:var(--txt2)">'+p.weight_pct.toFixed(1)+'%</span>'
      +(p.weight_pct>30?'<span style="font-size:7px;color:var(--dn)">HIGH</span>':'')
      +'</div>';
  }).join('');
  // Kelly sizing
  var kellyEl=document.getElementById('epf-risk-kelly');
  if(kellyEl) kellyEl.innerHTML=d.positions.filter(function(p){return p.ann_vol;}).map(function(p){
    var vol=(p.ann_vol||50)/100;
    var ret=(p.pnl_pct||0)/100;
    var kelly=vol>0?Math.max(0,Math.min((ret/(vol*vol))*100,100)).toFixed(1):'—';
    var drift=p.weight_pct-(parseFloat(kelly)||0);
    var driftCol=Math.abs(drift)>10?'var(--dn)':'var(--yel)';
    return '<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(46,46,46,.4);font-size:10px">'
      +'<span style="color:var(--cyn)">'+p.ticker+'</span>'
      +'<span style="color:var(--txt2)">Actual '+p.weight_pct.toFixed(1)+'%</span>'
      +'<span style="color:var(--yel)">Kelly ~'+kelly+'%</span>'
      +'<span style="color:'+driftCol+'">Drift '+(drift>=0?'+':'')+drift.toFixed(1)+'%</span>'
      +'</div>';
  }).join('')||'<span style="color:var(--txt3);font-size:10px">Need price history for Kelly sizing.</span>';
  // VaR panel
  var varEl=document.getElementById('epf-risk-var');
  if(varEl){
    if(_deepData&&_deepData.var95_pct!=null){
      varEl.innerHTML='<div style="font-size:20px;font-weight:700;color:var(--dn)">-'+_deepData.var95_pct.toFixed(2)+'%</div>'
        +'<div style="font-size:9px;color:var(--txt3);margin-bottom:8px">1-day VaR at 95% confidence</div>'
        +'<div style="font-size:16px;font-weight:700;color:var(--dn)">-'+(_deepData.cvar95_pct||0).toFixed(2)+'%</div>'
        +'<div style="font-size:9px;color:var(--txt3);margin-bottom:8px">CVaR / Expected Shortfall</div>'
        +(_deepData.var99_pct?'<div style="font-size:14px;font-weight:700;color:var(--dn)">-'+_deepData.var99_pct.toFixed(2)+'%</div><div style="font-size:9px;color:var(--txt3)">VaR 99%</div>':'');
    } else {
      varEl.innerHTML='<div style="color:var(--txt3);font-size:10px">Open ANALYTICS tab first to compute VaR.</div>';
    }
  }
  // Risk flags
  var flagsEl=document.getElementById('epf-risk-flags');
  var flags=[];
  if(s.concentration==='HIGH') flags.push({t:'HIGH CONCENTRATION',d:'HHI '+s.hhi.toFixed(0)+' — portfolio heavily concentrated. Consider diversifying.',c:'var(--dn)'});
  d.positions.forEach(function(p){
    if(p.weight_pct>30) flags.push({t:'OVERWEIGHT: '+p.ticker,d:p.weight_pct.toFixed(1)+'% of portfolio in single position.',c:'var(--yel)'});
    if(p.pnl_pct<-20) flags.push({t:'LARGE DRAWDOWN: '+p.ticker,d:p.pnl_pct.toFixed(2)+'% below entry. Review stop-loss.',c:'var(--dn)'});
    if(p.sharpe!=null&&p.sharpe<-1) flags.push({t:'POOR RISK-ADJUSTED RETURN: '+p.ticker,d:'Sharpe '+p.sharpe.toFixed(2)+' — poor risk/reward ratio.',c:'var(--yel)'});
    if(p.ann_vol!=null&&p.ann_vol>200) flags.push({t:'EXTREME VOLATILITY: '+p.ticker,d:'Annualised vol '+p.ann_vol.toFixed(1)+'% — very high risk.',c:'var(--dn)'});
  });
  if(d.cash/s.total_with_cash<0.05&&d.positions.length>0) flags.push({t:'LOW CASH BUFFER',d:'Only '+(d.cash/s.total_with_cash*100).toFixed(1)+'% cash.',c:'var(--yel)'});
  if(flagsEl) flagsEl.innerHTML=flags.length===0
    ?'<div style="color:var(--up);font-size:10px">✓ No major risk flags detected.</div>'
    :flags.map(function(fl){return '<div style="border-left:3px solid '+fl.c+';padding:6px 10px;background:rgba(0,0,0,.2);margin-bottom:6px"><div style="font-size:9px;font-weight:700;color:'+fl.c+';letter-spacing:1px;margin-bottom:2px">'+fl.t+'</div><div style="font-size:10px;color:var(--txt2)">'+fl.d+'</div></div>';}).join('');
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
function renderBenchmark(){
  if(!_bmData) return;
  var d=_bmData;
  // KPIs
  var kpis=document.getElementById('epf-bm-kpis');
  if(kpis){
    var outC=d.outperformance>=0?'var(--up)':'var(--dn)';
    kpis.innerHTML=[
      {l:'PORTFOLIO RETURN',v:(d.portfolio_return>=0?'+':'')+d.portfolio_return.toFixed(2)+'%',c:d.portfolio_return>=0?'var(--up)':'var(--dn)'},
      {l:'BENCHMARK RETURN',v:(d.benchmark_return>=0?'+':'')+d.benchmark_return.toFixed(2)+'%',c:d.benchmark_return>=0?'var(--up)':'var(--dn)'},
      {l:'ALPHA',v:d.alpha!=null?(d.alpha>=0?'+':'')+d.alpha.toFixed(2)+'%':'—',c:d.alpha>=0?'var(--up)':'var(--dn)'},
      {l:'BETA',v:d.beta!=null?d.beta.toFixed(3):'—',c:'var(--yel)'},
    ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');
  }
  // Chart — interactive with crosshair + shift-drag zoom
  var ctx=document.getElementById('epf-bm-chart');
  if(ctx){
    if(_charts.bm) _charts.bm.destroy();
    var n=Math.min(d.portfolio_curve.length,d.benchmark_curve.length);
    var pfCurve=d.portfolio_curve.slice(0,n);
    var bmCurve=d.benchmark_curve.slice(0,n);
    var labels=Array.from({length:n},function(_,i){return i;});
    var bmLbl=d.benchmark_label||'Benchmark';
    _charts.bm=new Chart(ctx,{
      type:'line',
      data:{labels:labels,datasets:[
        {label:'Portfolio',data:pfCurve,borderColor:'#ff8c00',borderWidth:2,
         pointRadius:0,pointHoverRadius:5,pointHoverBackgroundColor:'#ff8c00',fill:false},
        {label:bmLbl,data:bmCurve,borderColor:'#534AB7',borderWidth:1.5,
         pointRadius:0,pointHoverRadius:4,fill:false,borderDash:[5,3]}
      ]},
      options:{
        responsive:true,maintainAspectRatio:false,animation:false,
        interaction:{mode:'index',intersect:false},
        plugins:{
          legend:{labels:{color:'#888',font:{size:9},boxWidth:12,padding:12}},
          tooltip:{
            backgroundColor:'rgba(15,15,15,0.92)',
            borderColor:'rgba(83,74,183,0.5)',
            borderWidth:1,
            titleColor:'#aaa',
            bodyColor:'#ccc',
            padding:10,
            callbacks:{
              title:function(items){return 'Day '+items[0].dataIndex;},
              label:function(c){
                var v=c.parsed.y;
                var ret=(v/10000-1)*100;
                var col=c.datasetIndex===0?'#ff8c00':'#534AB7';
                return c.dataset.label+':  $'+f(v,2)+'  ('+(ret>=0?'+':'')+ret.toFixed(2)+'%)';
              },
              afterBody:function(items){
                if(items.length<2) return '';
                var pf=items[0].parsed.y; var bm=items[1].parsed.y;
                var diff=(pf-bm)/bm*100;
                return ['','  Spread vs BM: '+(diff>=0?'+':'')+diff.toFixed(2)+'%'];
              }
            }
          }
        },
        scales:{
          x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#555',maxTicksLimit:10,font:{size:9}}},
          y:{grid:{color:'rgba(30,30,30,.8)'},position:'right',
             ticks:{color:'#555',font:{size:9},callback:function(v){return '$'+fk(v);}}}
        }
      },
      plugins:[_crosshairPlugin]
    });
    // Shift+drag to zoom, dblclick to reset
    _initChartZoom(ctx, _charts.bm, pfCurve, labels);
  }
  // Detail
  var det=document.getElementById('epf-bm-detail');
  if(det){
    var rows=[
      ['Portfolio Return', (d.portfolio_return>=0?'+':'')+d.portfolio_return.toFixed(2)+'%'],
      ['Benchmark (NER EW)', (d.benchmark_return>=0?'+':'')+d.benchmark_return.toFixed(2)+'%'],
      ['Outperformance', (d.outperformance>=0?'+':'')+d.outperformance.toFixed(2)+'%'],
      ['Alpha', d.alpha!=null?(d.alpha>=0?'+':'')+d.alpha.toFixed(2)+'%':'—'],
      ['Beta', d.beta!=null?d.beta.toFixed(3):'—'],
      ['Benchmark Tickers', d.benchmark_tickers.join(', ')],
    ];
    det.innerHTML=rows.map(function(r){
      return '<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(46,46,46,.3);font-size:10px">'
        +'<span style="color:var(--txt2)">'+r[0]+'</span><span style="font-weight:700;color:var(--wht)">'+r[1]+'</span></div>';
    }).join('');
  }
}

// ── Attribution ───────────────────────────────────────────────────────────────
function renderAttribution(){
  if(!_deepData||!_deepData.attribution.length) return;
  var ctx=document.getElementById('epf-attr-chart');
  if(ctx){
    if(_charts.attr) _charts.attr.destroy();
    var items=_deepData.attribution;
    _charts.attr=new Chart(ctx,{
      type:'bar',
      data:{labels:items.map(function(a){return a.ticker;}),datasets:[{data:items.map(function(a){return a.contribution_pct;}),backgroundColor:items.map(function(a){return a.contribution_pct>=0?'rgba(0,200,83,.6)':'rgba(244,67,54,.6)';}),borderWidth:0}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,indexAxis:'y',plugins:{legend:{display:false},tooltip:{...TT}},scales:{x:{grid:{color:'#1a1a1a'},ticks:{color:'#444',callback:function(v){return v.toFixed(2)+'%';}}},y:{grid:{color:'#1a1a1a'},ticks:{color:'#888'}}}}});
  }
  var tbl=document.getElementById('epf-attr-table');
  if(tbl){
    tbl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th class="r">RETURN%</th><th class="r">WEIGHT%</th><th class="r">CONTRIBUTION%</th><th class="r">ANN VOL</th><th class="r">SHARPE</th></tr></thead><tbody>'
      +_deepData.attribution.map(function(a){
        var cc=a.contribution_pct>=0?'var(--up)':'var(--dn)';
        return '<tr><td style="color:var(--cyn);font-weight:700">'+a.ticker+'</td>'
          +'<td class="r" style="color:'+(a.return_pct>=0?'var(--up)':'var(--dn)')+'">'+(a.return_pct>=0?'+':'')+a.return_pct.toFixed(2)+'%</td>'
          +'<td class="r">'+a.weight_pct.toFixed(1)+'%</td>'
          +'<td class="r" style="color:'+cc+';font-weight:700">'+(a.contribution_pct>=0?'+':'')+a.contribution_pct.toFixed(3)+'%</td>'
          +'<td class="r" style="color:var(--yel)">'+(a.ann_vol!=null?a.ann_vol.toFixed(1)+'%':'—')+'</td>'
          +'<td class="r" style="color:'+(a.sharpe>0?'var(--up)':'var(--dn)')+'">'+(a.sharpe!=null?a.sharpe.toFixed(2):'—')+'</td>'
          +'</tr>';
      }).join('')+'</tbody></table>';
  }
}

// ── Stress Tests ──────────────────────────────────────────────────────────────
function renderStressTests(){
  if(!_deepData||!_deepData.stress_tests) return;
  _renderStressData(_deepData.stress_tests);
}

function _renderStressData(scenarios){
  var ctx=document.getElementById('epf-stress-chart');
  if(ctx){
    if(_charts.stress) _charts.stress.destroy();
    _charts.stress=new Chart(ctx,{
      type:'bar',
      data:{labels:scenarios.map(function(s){return s.name;}),datasets:[{data:scenarios.map(function(s){return s.portfolio_impact_pct;}),backgroundColor:scenarios.map(function(s){return s.portfolio_impact_pct>=0?'rgba(0,200,83,.6)':'rgba(244,67,54,.6)';}),borderWidth:0}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,indexAxis:'y',plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return ' Portfolio impact: '+(c.parsed.x>=0?'+':'')+c.parsed.x.toFixed(2)+'%';}}}},scales:{x:{grid:{color:'#1a1a1a'},ticks:{color:'#444',callback:function(v){return v.toFixed(1)+'%';}}},y:{grid:{color:'#1a1a1a'},ticks:{color:'#888'}}}}});
  }
  var tbl=document.getElementById('epf-stress-table');
  if(tbl){
    tbl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>SCENARIO</th><th class="r">PORTFOLIO IMPACT</th><th class="r">EST. LOSS ($)</th></tr></thead><tbody>'
      +scenarios.map(function(s){
        var aum=_pfData&&_pfData.summary.total_with_cash||0;
        var lossVal=aum*s.portfolio_impact_pct/100;
        var col=s.portfolio_impact_pct>=0?'var(--up)':'var(--dn)';
        return '<tr><td>'+s.name+'</td>'
          +'<td class="r" style="color:'+col+';font-weight:700">'+(s.portfolio_impact_pct>=0?'+':'')+s.portfolio_impact_pct.toFixed(2)+'%</td>'
          +'<td class="r" style="color:'+col+'">'+(lossVal>=0?'+':'')+'$'+f(lossVal,2)+'</td>'
          +'</tr>';
      }).join('')+'</tbody></table>';
  }
}

window.runCustomStress = function(){
  if(!_pfData) return;
  var ticker=(document.getElementById('stress-ticker').value||'').trim().toUpperCase();
  var shock=parseFloat(document.getElementById('stress-shock').value)/100;
  if(!ticker||isNaN(shock)){alert('Enter ticker and shock %');return;}
  var pos=_pfData.positions.find(function(p){return p.ticker===ticker;});
  if(!pos){alert('Ticker not in portfolio');return;}
  var impact=pos.weight_pct/100*shock*100;
  var aum=_pfData.summary.total_with_cash;
  var lossVal=aum*impact/100;
  var col=impact>=0?'var(--up)':'var(--dn)';
  var el=document.getElementById('epf-custom-stress-result');
  if(el) el.innerHTML='<div style="font-size:11px;padding:8px 0">'
    +'<span style="color:var(--cyn)">'+ticker+'</span> shock of '
    +'<span style="color:'+col+';">'+(shock>=0?'+':'')+(shock*100).toFixed(1)+'%</span>'
    +' → portfolio impact: <span style="font-weight:700;color:'+col+'">'+(impact>=0?'+':'')+impact.toFixed(2)+'%</span>'
    +' (<span style="color:'+col+'">'+(lossVal>=0?'+':'')+'$'+f(lossVal,2)+'</span>)'
    +'</div>';
};

// ── Realized P&L ──────────────────────────────────────────────────────────────
function renderRealized(){
  var realPnl=_pfData&&_pfData.realized_pnl||0;
  var kpis=document.getElementById('epf-real-kpis');
  if(kpis){
    var deepReal=(_deepData&&_deepData.realized_pnl!=null?_deepData.realized_pnl:null)||realPnl||0;
    var divIncome=(_pfData&&_pfData.dividend_income)||0;
    var trades=(_deepData&&_deepData.realized_trades)||[];
    var winners=trades.filter(function(t){return (t.realised_pnl||0)>0;}).length;
    kpis.innerHTML=[
      {l:'REALIZED P&L',v:(deepReal>=0?'+':'')+'$'+f(Math.abs(deepReal),2),c:deepReal>=0?'var(--up)':'var(--dn)'},
      {l:'DIVIDEND INCOME',v:'+$'+f(divIncome,2),c:'var(--up)'},
      {l:'CLOSED TRADES',v:trades.length,c:'var(--wht)'},
      {l:'WIN RATE',v:trades.length?Math.round(winners/trades.length*100)+'%':'—',c:'var(--yel)'},
      {l:'TOTAL INCOME',v:'+'+'$'+f(Math.abs(deepReal+divIncome),2),c:'var(--up)'},
    ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');
  }
  // From deep analytics if available, else from position.closes
  var trades2=[];
  if(_deepData&&_deepData.realized_trades) {
    trades2=_deepData.realized_trades;
  } else {
    (_pfData&&_pfData.positions||[]).forEach(function(p){
      (p.closes||[]).forEach(function(c){
        trades2.push({ticker:p.ticker,close_price:c.close_price,close_qty:c.close_qty,close_date:c.close_date,realised_pnl:c.realised_pnl,notes:c.notes});
      });
    });
  }
  var tbl=document.getElementById('epf-realized-table');
  if(tbl){
    if(!trades2.length){
      tbl.innerHTML='<div style="color:var(--txt3);font-size:10px;padding:10px">No closed trades yet. Use CLOSE POSITION to record realized gains/losses.</div>';
    } else {
      tbl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th>DATE</th><th class="r">QTY</th><th class="r">CLOSE PRICE</th><th class="r">REALISED P&L</th><th>NOTES</th></tr></thead><tbody>'
        +trades2.map(function(t){
          var pc=(t.realised_pnl||0)>=0?'var(--up)':'var(--dn)';
          return '<tr><td style="color:var(--cyn);font-weight:700">'+(t.ticker||'?')+'</td>'
            +'<td>'+(t.close_date||'—')+'</td>'
            +'<td class="r">'+(t.close_qty||'—')+'</td>'
            +'<td class="r">$'+f(t.close_price||0,4)+'</td>'
            +'<td class="r" style="color:'+pc+';font-weight:700">'+(t.realised_pnl>=0?'+':'')+'$'+f(t.realised_pnl||0,2)+'</td>'
            +'<td>'+(t.notes||'')+'</td></tr>';
        }).join('')+'</tbody></table>';
    }
  }
}

// ── Target Allocation ─────────────────────────────────────────────────────────
window.renderTargetAlloc = async function(){
  if(!_pfData) return;
  var r=await api('/api/enterprise/portfolios/'+_pfid+'/target_allocation');
  var targets=r.ok?r.d.targets:{};
  var el=document.getElementById('epf-target-alloc');
  if(!el) return;
  el.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th class="r">ACTUAL %</th><th class="r">TARGET %</th><th>DRIFT</th><th class="r">SET TARGET</th></tr></thead><tbody>'
    +_pfData.positions.map(function(p){
      var tgt=targets[p.ticker]||0;
      var drift=p.weight_pct-tgt;
      var dCol=Math.abs(drift)>5?'var(--dn)':Math.abs(drift)>2?'var(--yel)':'var(--up)';
      return '<tr><td style="color:var(--cyn);font-weight:700">'+p.ticker+'</td>'
        +'<td class="r">'+p.weight_pct.toFixed(1)+'%</td>'
        +'<td class="r" style="color:var(--yel)">'+tgt.toFixed(1)+'%</td>'
        +'<td><span style="color:'+dCol+'">'+(drift>=0?'+':'')+drift.toFixed(1)+'%</span></td>'
        +'<td class="r"><input type="number" value="'+tgt+'" step="0.5" min="0" max="100" style="width:70px;background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:2px 4px;outline:none" oninput="_targetsEdited[\''+p.ticker+'\']=parseFloat(this.value)||0"></td>'
        +'</tr>';
    }).join('')+'</tbody></table>';

  // Drift chart
  var ctx=document.getElementById('epf-drift-chart');
  if(ctx){
    if(_charts.drift) _charts.drift.destroy();
    var items=_pfData.positions.map(function(p){return {t:p.ticker,a:p.weight_pct,tg:targets[p.ticker]||0};});
    _charts.drift=new Chart(ctx,{
      type:'bar',
      data:{labels:items.map(function(i){return i.t;}),datasets:[
        {label:'Actual',data:items.map(function(i){return i.a;}),backgroundColor:'rgba(0,200,83,.6)',borderWidth:0},
        {label:'Target',data:items.map(function(i){return i.tg;}),backgroundColor:'rgba(255,140,0,.4)',borderWidth:0}
      ]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:'#888',font:{size:9}}},tooltip:{...TT}},scales:{x:{grid:{color:'#1a1a1a'},ticks:{color:'#444'}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',callback:function(v){return v.toFixed(0)+'%';}}}}}});
  }

  // Rebalance suggestions
  var reb=document.getElementById('epf-rebalance');
  if(reb){
    var totalAUM=_pfData.summary.total_with_cash;
    var suggestions=_pfData.positions.map(function(p){
      var tgt=(targets[p.ticker]||0)/100;
      var actual=p.weight_pct/100;
      var diff=tgt-actual;
      var valDiff=diff*totalAUM;
      var lp=p.live_price||p.entry_price;
      var shares=lp>0?Math.round(Math.abs(valDiff)/lp):0;
      return {ticker:p.ticker,diff:diff,valDiff:valDiff,shares:shares};
    }).filter(function(s){return Math.abs(s.diff)>0.02;})
      .sort(function(a,b){return Math.abs(b.diff)-Math.abs(a.diff);});
    if(!suggestions.length){
      reb.innerHTML='<div style="color:var(--up);font-size:10px">✓ Portfolio is within 2% of all targets.</div>';
    } else {
      reb.innerHTML=suggestions.map(function(s){
        var action=s.diff>0?'BUY':'SELL';
        var col=s.diff>0?'var(--up)':'var(--dn)';
        return '<div style="display:flex;align-items:center;gap:12px;padding:5px 0;border-bottom:1px solid rgba(46,46,46,.3);font-size:10px">'
          +'<span style="color:var(--cyn);font-weight:700;width:60px">'+s.ticker+'</span>'
          +'<span style="color:'+col+';font-weight:700;width:40px">'+action+'</span>'
          +'<span>~'+s.shares+' shares</span>'
          +'<span style="color:var(--txt2)">($'+(Math.abs(s.valDiff)).toFixed(0)+')</span>'
          +'<span style="color:var(--txt3)">'+(s.diff>=0?'+':'')+( s.diff*100).toFixed(1)+'% drift</span>'
          +'</div>';
      }).join('');
    }
  }
}

window.saveTargetAllocation = async function(){
  if(!_pfid) return;
  var targets={};
  _pfData&&_pfData.positions.forEach(function(p){
    var el=document.querySelector('input[oninput*="'+p.ticker+'"]');
    if(el) targets[p.ticker]=parseFloat(el.value)||0;
  });
  Object.assign(targets,_targetsEdited);
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/target_allocation',{targets:targets});
  if(r.ok){ var bar=document.getElementById('cmd-st');if(bar){bar.textContent='Target allocation saved';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);} }
  await renderTargetAlloc();
};

// ── Cash log ──────────────────────────────────────────────────────────────────
function renderCashLog(){
  var tbody=document.getElementById('epf-cash-tbody');
  if(!tbody) return;
  var log=(_pfData&&_pfData.cash_log||[]).slice().reverse();
  var fromStr=(document.getElementById('cl-date-from')||{}).value||'';
  var toStr=(document.getElementById('cl-date-to')||{}).value||'';
  if(fromStr) log=log.filter(function(r){return !r.ts||(r.ts.slice(0,10)>=fromStr);});
  if(toStr)   log=log.filter(function(r){return !r.ts||(r.ts.slice(0,10)<=toStr);});
  tbody.innerHTML=log.map(function(r){
    return '<tr><td style="color:var(--txt2)">'+(r.ts?new Date(r.ts).toLocaleString('en-GB'):'—')+'</td>'
      +'<td style="color:'+(r.amount>=0?'var(--up)':'var(--dn)')+';font-weight:700">'+(r.amount>=0?'+':'')+'$'+f(r.amount,2)+'</td>'
      +'<td>'+(r.note||'—')+'</td>'
      +'<td class="r" style="color:var(--wht)">$'+f(r.balance_after||0,2)+'</td>'
      +'</tr>';
  }).join('')||'<tr><td colspan="4" style="padding:16px;color:var(--txt3);text-align:center">No entries for selected period.</td></tr>';
}

// ── Audit log ─────────────────────────────────────────────────────────────────
window.renderAuditLog = async function(){
  var tbody=document.getElementById('epf-audit-tbody');
  if(!tbody) return;
  var r=await api('/api/enterprise/portfolios/'+(_pfid||'')+'/audit_log');
  var log=r.ok?r.d:[];
  var fromStr=(document.getElementById('al-date-from')||{}).value||'';
  var toStr=(document.getElementById('al-date-to')||{}).value||'';
  if(fromStr) log=log.filter(function(e){return !e.ts||(e.ts.slice(0,10)>=fromStr);});
  if(toStr)   log=log.filter(function(e){return !e.ts||(e.ts.slice(0,10)<=toStr);});
  tbody.innerHTML=log.slice().reverse().map(function(e){
    return '<tr><td style="color:var(--txt2);font-size:9px">'+(e.ts?new Date(e.ts).toLocaleString('en-GB'):'—')+'</td>'
      +'<td style="color:var(--org);font-weight:700">'+e.action+'</td>'
      +'<td style="color:var(--txt2)">'+(e.detail||'')+'</td></tr>';
  }).join('')||'<tr><td colspan="3" style="padding:16px;color:var(--txt3);text-align:center">No audit events for selected period.</td></tr>';
}

// ── Notes save ────────────────────────────────────────────────────────────────
window.saveNotes = async function(){
  if(!_pfid||!_pfData) return;
  var notes=(document.getElementById('epf-notes-area')||{}).value||'';
  var strategy=(document.getElementById('epf-strategy-input')||{}).value||'';
  var pf=ENT.portfolios.find(function(p){return p.id===_pfid;})||{};
  await apiPost('/api/enterprise/portfolios',Object.assign({},pf,{id:_pfid,notes:notes,strategy:strategy}));
  var bar=document.getElementById('cmd-st');
  if(bar){bar.textContent='Notes saved';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},1500);}
};

// ── Position / Close modals ───────────────────────────────────────────────────
function _moveToBody(id){var m=document.getElementById(id);if(m&&m.parentElement!==document.body)document.body.appendChild(m);}
window.showAddPosModal = function(){_moveToBody('add-pos-modal');var m=document.getElementById('add-pos-modal');if(m)m.style.display='flex';};
window.hideAddPosModal = function(){var m=document.getElementById('add-pos-modal');if(m)m.style.display='none';};
window.showCashModal = function(sign){_moveToBody('cash-modal');var m=document.getElementById('cash-modal');if(m)m.style.display='flex';var inp=document.getElementById('cm-amount');if(sign&&inp)inp.value=sign>0?'':'-';};
window.hideCashModal = function(){var m=document.getElementById('cash-modal');if(m)m.style.display='none';};
window.showImportModal = function(){_moveToBody('import-modal');document.getElementById('import-raw').value='';document.getElementById('import-preview').style.display='none';document.getElementById('import-status').style.display='none';var btn=document.getElementById('import-confirm-btn');if(btn)btn.style.display='none';_importParsed=[];var m=document.getElementById('import-modal');if(m)m.style.display='flex';};
window.hideImportModal = function(){var m=document.getElementById('import-modal');if(m)m.style.display='none';};

window.toggleCloseMode = function(val){
  var isClose=val==='close';
  var cf=document.getElementById('close-pos-fields');if(cf)cf.style.display=isClose?'block':'none';
  var btn=document.getElementById('add-pos-btn');if(btn)btn.textContent=isClose?'CLOSE POSITION \u2192':'ADD POSITION \u2192';
  ['apm-ticker','apm-qty','apm-price','apm-date'].forEach(function(id){var el=document.getElementById(id);if(el){var row=el.closest('.prow');if(row)row.style.display=isClose?'none':'flex';}});
  if(isClose){
    var sel=document.getElementById('close-pos-select');
    if(sel&&_pfData){
      sel.innerHTML='<option value="">— select —</option>'+_pfData.positions.map(function(p){return '<option value="'+p.id+'">'+p.ticker+' — '+p.qty.toLocaleString()+' @ $'+f(p.entry_price,4)+'</option>';}).join('');
    }
    var dt=document.getElementById('close-date');if(dt&&!dt.value)dt.value=new Date().toISOString().slice(0,10);
  }
};

window.prefillCloseFields = function(posId){
  if(!posId||!_pfData) return;
  var pos=_pfData.positions.find(function(p){return p.id===posId;});
  if(!pos) return;
  var q=document.getElementById('close-qty');if(q)q.value=pos.qty;
  var pr=document.getElementById('close-price');if(pr)pr.value=pos.live_price!=null?pos.live_price:pos.entry_price;
  updateClosePnlPreview();
};

window.updateClosePnlPreview = function(){
  var sel=document.getElementById('close-pos-select');
  var price=parseFloat(document.getElementById('close-price').value);
  var qty=parseFloat(document.getElementById('close-qty').value);
  var el=document.getElementById('close-pnl-val');
  if(!el||!sel||!_pfData){return;}
  var pos=_pfData.positions.find(function(p){return p.id===sel.value;});
  if(!pos||isNaN(price)||isNaN(qty)){el.textContent='—';el.style.color='var(--txt2)';return;}
  var pnl=(price-pos.entry_price)*qty;
  el.textContent=(pnl>=0?'+':'')+'$'+f(pnl,2);
  el.style.color=pnl>=0?'var(--up)':'var(--dn)';
};

window.addPosition = async function(){
  var type=document.getElementById('apm-type').value;
  if(type==='close'){
    var posId=document.getElementById('close-pos-select').value;
    var closePrice=parseFloat(document.getElementById('close-price').value);
    var closeQty=parseFloat(document.getElementById('close-qty').value)||null;
    var closeDate=document.getElementById('close-date').value;
    var notes=document.getElementById('apm-notes').value;
    if(!posId){alert('Select a position to close.');return;}
    if(!closePrice||closePrice<=0){alert('Enter a valid close price.');return;}
    var pos=_pfData&&_pfData.positions.find(function(p){return p.id===posId;});
    var ticker=pos?pos.ticker:'?';
    var body={close_price:closePrice,close_date:closeDate,notes:notes};
    if(closeQty)body.close_qty=closeQty;
    var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId+'/close',body);
    if(!r.ok){alert('Error closing: '+(r.d.detail||'unknown'));return;}
    hideAddPosModal();
    document.getElementById('apm-type').value='long';
    toggleCloseMode('long');
    document.getElementById('apm-notes').value='';
    await ENT_PF_refresh();
    epfTab('realized');
    var pnl=r.d.realised_pnl;
    var bar=document.getElementById('cmd-st');
    if(bar){bar.textContent=(r.d.removed?'CLOSED ':'PARTIAL CLOSE ')+ticker+' | PnL: '+(pnl>=0?'+':'')+'$'+f(pnl,2);bar.style.color=pnl>=0?'var(--up)':'var(--dn)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},4000);}
    return;
  }
  var ticker=(document.getElementById('apm-ticker').value||'').trim().toUpperCase();
  var qty=parseFloat(document.getElementById('apm-qty').value)||0;
  var price=parseFloat(document.getElementById('apm-price').value)||0;
  var date=document.getElementById('apm-date').value;
  var notes=document.getElementById('apm-notes').value;
  if(!ticker||!qty||!price){alert('Ticker, quantity and price required.');return;}
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions',{ticker:ticker,qty:qty,entry_price:price,entry_date:date,notes:notes,type:type});
  if(!r.ok){alert('Error adding position');return;}
  hideAddPosModal();
  ['apm-ticker','apm-qty','apm-price','apm-notes'].forEach(function(id){document.getElementById(id).value='';});
  await ENT_PF_refresh();
  epfTab('positions');
};

window.removePosition = async function(posId,ticker){
  if(!confirm('Remove '+ticker+' from portfolio?'))return;
  await api('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId,{method:'DELETE'});
  await ENT_PF_refresh();
};

window.adjustCash = async function(){
  var amount=parseFloat(document.getElementById('cm-amount').value);
  var note=document.getElementById('cm-note').value;
  if(isNaN(amount)){alert('Enter a valid amount');return;}
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/cash',{amount:amount,note:note});
  if(!r.ok){alert('Error updating cash');return;}
  hideCashModal();
  document.getElementById('cm-amount').value='';document.getElementById('cm-note').value='';
  await ENT_PF_refresh();epfTab('cashlog');
};

// ── Import ────────────────────────────────────────────────────────────────────
var _importParsed=[];
window.previewImport = function(){
  var raw=(document.getElementById('import-raw').value||'').trim();
  if(!raw){alert('Paste the NER table first.');return;}
  _importParsed=[];
  var lines=raw.split('\n');
  for(var i=0;i<lines.length;i++){
    var trimmed=lines[i].trim();
    if(!trimmed.startsWith('|'))continue;
    var parts=trimmed.split('|').map(function(p){return p.trim();}).filter(function(p){return p.length>0;});
    if(parts.length<3)continue;
    var ticker=parts[0].toUpperCase();
    if(ticker==='TICKER'||ticker==='---'||ticker===''||ticker.startsWith('-')||ticker.startsWith('+'))continue;
    var qty=parseFloat(parts[1].replace(/,/g,''));
    var avgCost=parseFloat(parts[2].replace(/[$,]/g,'').trim());
    if(isNaN(qty)||isNaN(avgCost)||qty<=0||avgCost<=0)continue;
    _importParsed.push({ticker:ticker,qty:qty,entry_price:avgCost});
  }
  if(_importParsed.length===0){var st=document.getElementById('import-status');st.style.display='block';st.style.color='var(--dn)';st.textContent='Could not parse any positions.';return;}
  var existing=new Set((_pfData?_pfData.positions:[]).map(function(p){return p.ticker;}));
  var dupes=_importParsed.filter(function(p){return existing.has(p.ticker);});
  document.getElementById('import-preview-rows').innerHTML=_importParsed.map(function(p){
    var isDupe=existing.has(p.ticker);
    return '<div style="'+(isDupe?'color:var(--txt3);text-decoration:line-through':'color:var(--wht)')+';padding:2px 0">'+p.ticker.padEnd(8)+'  '+String(p.qty).padStart(6)+'  @ $'+f(p.entry_price,4)+(isDupe?' <span style="color:var(--yel);font-size:8px">SKIP</span>':'')+'</div>';
  }).join('');
  document.getElementById('import-preview').style.display='block';
  var toImport=_importParsed.filter(function(p){return !existing.has(p.ticker);});
  var st=document.getElementById('import-status');st.style.display='block';st.style.color=toImport.length>0?'var(--up)':'var(--yel)';st.textContent=toImport.length+' to import'+(dupes.length>0?', '+dupes.length+' skipped':'');
  var btn=document.getElementById('import-confirm-btn');if(btn)btn.style.display=toImport.length>0?'inline-block':'none';
};
window.confirmImport = async function(){
  if(!_importParsed.length){alert('Nothing to import.');return;}
  var btn=document.getElementById('import-confirm-btn');if(btn){btn.disabled=true;btn.textContent='IMPORTING\u2026';}
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/import',{positions:_importParsed.map(function(p){return {ticker:p.ticker,qty:p.qty,entry_price:p.entry_price,entry_date:'',notes:'Imported from NER terminal',type:'long'};})});
  if(btn){btn.disabled=false;btn.textContent='CONFIRM IMPORT \u2192';}
  if(!r.ok){alert('Import failed: '+(r.d.detail||'unknown'));return;}
  hideImportModal();
  await ENT_PF_refresh();epfTab('positions');
  var bar=document.getElementById('cmd-st');if(bar){bar.textContent='Imported '+r.d.imported+' positions'+(r.d.skipped>0?' ('+r.d.skipped+' skipped)':'');bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},4000);}
};

// ── Report ────────────────────────────────────────────────────────────────────
function _rptSec(id){var el=document.getElementById(id);return el&&el.checked;}
function _rptH(title,color){return '<div style="font-size:9px;font-weight:700;letter-spacing:2px;color:'+(color||'var(--org)')+';margin:20px 0 8px;padding-bottom:4px;border-bottom:1px solid rgba(255,140,0,.2)">'+title+'</div>';}
function _rptKV(label,value,vc){return '<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(46,46,46,.3);font-size:10px"><span style="color:var(--txt2)">'+label+'</span><span style="font-weight:700;color:'+(vc||'var(--wht)')+'">'+value+'</span></div>';}
function _rptTable(headers,rows,aligns){
  var ths=headers.map(function(h,i){var a=aligns&&aligns[i]?aligns[i]:'left';return '<th style="padding:4px 6px;text-align:'+a+';color:var(--txt2);border-bottom:1px solid var(--bdr);white-space:nowrap">'+h+'</th>';}).join('');
  var trs=rows.map(function(row){var tds=row.map(function(cell,i){var a=aligns&&aligns[i]?aligns[i]:'left';var v=typeof cell==='object'?cell.v:cell;var c=typeof cell==='object'?cell.c:'';return '<td style="padding:4px 6px;text-align:'+a+';'+(c?'color:'+c+';':'')+'">'+(v||'—')+'</td>';}).join('');return '<tr style="border-bottom:1px solid rgba(46,46,46,.25)">'+tds+'</tr>';}).join('');
  return '<table style="width:100%;border-collapse:collapse;font-size:10px;margin-bottom:10px"><thead><tr>'+ths+'</tr></thead><tbody>'+trs+'</tbody></table>';
}

window.generateReport = function(){
  if(!_pfData) return;
  var d=_pfData,s=d.summary;
  var date=new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'});
  var firm=(document.getElementById('rpt-firm-name')||{}).value||'BLOOMBERG / NER ENTERPRISE';
  var period=(document.getElementById('rpt-period')||{}).value||date;
  var el=document.getElementById('epf-report-body');
  if(!el) return;
  var html='<div style="font-family:\'Courier New\',monospace;color:var(--txt);max-width:900px">';

  if(_rptSec('rpt-cover')){
    html+='<div style="text-align:center;padding:32px 0 24px;border-bottom:2px solid var(--org);margin-bottom:24px">'
      +'<div style="font-size:9px;color:var(--txt3);letter-spacing:4px;margin-bottom:8px">'+firm.toUpperCase()+'</div>'
      +'<div style="font-size:28px;font-weight:700;color:var(--wht);letter-spacing:2px">PORTFOLIO STATEMENT</div>'
      +'<div style="font-size:11px;color:var(--txt2);margin-top:4px">Period: '+period+'</div>'
      +'<div style="display:inline-grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1px;background:var(--bdr);border:1px solid var(--bdr);margin-top:16px">'
      +'<div style="background:var(--bg3);padding:10px 16px"><div style="font-size:8px;color:var(--txt3)">CLIENT</div><div style="font-weight:700;color:var(--org)">'+(d.client||'—')+'</div></div>'
      +'<div style="background:var(--bg3);padding:10px 16px"><div style="font-size:8px;color:var(--txt3)">PORTFOLIO</div><div style="font-weight:700;color:var(--wht)">'+(d.name||'—')+'</div></div>'
      +'<div style="background:var(--bg3);padding:10px 16px"><div style="font-size:8px;color:var(--txt3)">STRATEGY</div><div style="font-weight:700;color:var(--txt2)">'+(d.strategy||'—')+'</div></div>'
      +'<div style="background:var(--bg3);padding:10px 16px"><div style="font-size:8px;color:var(--txt3)">TOTAL AUM</div><div style="font-weight:700;color:var(--wht)">$'+fk(s.total_with_cash)+'</div></div>'
      +'</div></div>';
  }
  if(_rptSec('rpt-summary')){
    var pnlC=s.total_pnl>=0?'var(--up)':'var(--dn)';
    var retC=s.total_pnl_pct>=0?'var(--up)':'var(--dn)';
    var concC=s.concentration==='HIGH'?'var(--dn)':s.concentration==='MEDIUM'?'var(--yel)':'var(--up)';
    var pm=_deepData&&_deepData.portfolio_metrics;
    html+=_rptH('EXECUTIVE SUMMARY');
    html+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:12px"><div>';
    html+=_rptKV('Report Date',date);html+=_rptKV('Period',period);
    html+=_rptKV('Client',d.client||'—');html+=_rptKV('Portfolio',d.name||'—');
    if(d.strategy)html+=_rptKV('Strategy',d.strategy);
    html+=_rptKV('Positions',s.num_positions);
    html+='</div><div>';
    html+=_rptKV('Market Value','$'+fk(s.total_value));
    html+=_rptKV('Cash','$'+fk(d.cash),'var(--cyn)');
    html+=_rptKV('Total AUM','$'+fk(s.total_with_cash),'var(--wht)');
    html+=_rptKV('Cost Basis','$'+fk(s.total_cost));
    html+=_rptKV('Unrealised P&L',(s.total_pnl>=0?'+':'')+'$'+fk(Math.abs(s.total_pnl))+' ('+s.total_pnl_pct.toFixed(2)+'%)',pnlC);
    if(d.realized_pnl)html+=_rptKV('Realised P&L',(d.realized_pnl>=0?'+':'')+'$'+f(Math.abs(d.realized_pnl||0),2),(d.realized_pnl||0)>=0?'var(--up)':'var(--dn)');
    html+='</div></div>';
    var avgVol=d.positions.filter(function(p){return p.ann_vol;}).reduce(function(a,p){return a+p.ann_vol;},0)/Math.max(d.positions.filter(function(p){return p.ann_vol;}).length,1);
    var avgShr=d.positions.filter(function(p){return p.sharpe;}).reduce(function(a,p){return a+p.sharpe;},0)/Math.max(d.positions.filter(function(p){return p.sharpe;}).length,1);
    html+='<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--bdr);border:1px solid var(--bdr);margin-bottom:12px">';
    var kpis2=[
      ['RETURN',(s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%',retC],
      ['TWR',pm?(pm.twr>=0?'+':'')+pm.twr.toFixed(2)+'%':'—',pm&&pm.twr>=0?'var(--up)':'var(--dn)'],
      ['SHARPE',pm?pm.sharpe.toFixed(2):'—',pm&&pm.sharpe>0?'var(--up)':'var(--dn)'],
      ['SORTINO',pm?pm.sortino.toFixed(2):'—',pm&&pm.sortino>0?'var(--up)':'var(--dn)'],
      ['AVG VOL',isNaN(avgVol)?'—':avgVol.toFixed(1)+'%','var(--yel)'],
      ['HHI',s.hhi.toFixed(0),concC]
    ];
    kpis2.forEach(function(k){html+='<div style="background:var(--bg3);padding:8px;text-align:center"><div style="font-size:8px;color:var(--txt3);margin-bottom:3px">'+k[0]+'</div><div style="font-size:14px;font-weight:700;color:'+k[2]+'">'+k[1]+'</div></div>';});
    html+='</div>';
  }
  if(_rptSec('rpt-holdings')){
    html+=_rptH('HOLDINGS DETAIL');
    html+=_rptTable(['TICKER','TYPE','QTY','ENTRY','LIVE','MKT VALUE','P&L','P&L%','WEIGHT','ANN VOL','SHARPE','ENTRY DATE'],
      d.positions.map(function(p){
        var pc=p.pnl>=0?'var(--up)':'var(--dn)';var ppc=p.pnl_pct>=0?'var(--up)':'var(--dn)';
        return [{v:p.ticker,c:'var(--cyn)'},{v:(p.type||'long').toUpperCase()},p.qty.toLocaleString(),'$'+f(p.entry_price,4),{v:p.live_price!=null?'$'+f(p.live_price,4):'—'},'$'+fk(p.market_value),{v:(p.pnl>=0?'+':'')+'$'+f(p.pnl,2),c:pc},{v:(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%',c:ppc},p.weight_pct.toFixed(1)+'%',{v:p.ann_vol!=null?p.ann_vol.toFixed(1)+'%':'—'},{v:p.sharpe!=null?p.sharpe.toFixed(2):'—'},p.entry_date||'—'];
      }),['left','left','right','right','right','right','right','right','right','right','right','left']);
  }
  if(_rptSec('rpt-perf')){
    html+=_rptH('PERFORMANCE ANALYSIS');
    var sorted_pnl=d.positions.slice().sort(function(a,b){return b.pnl_pct-a.pnl_pct;});
    html+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px"><div>'+_rptH('TOP PERFORMERS','var(--up)');
    sorted_pnl.slice(0,3).forEach(function(p){var w=Math.min(Math.abs(p.pnl_pct),100).toFixed(0);html+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px"><span style="width:52px;color:var(--cyn);font-weight:700">'+p.ticker+'</span><div style="flex:1;height:5px;background:var(--bdr)"><div style="width:'+w+'%;height:100%;background:var(--up)"></div></div><span style="color:var(--up);width:60px;text-align:right">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</span></div>';});
    html+='</div><div>'+_rptH('UNDERPERFORMERS','var(--dn)');
    sorted_pnl.slice().reverse().slice(0,3).filter(function(p){return p.pnl_pct<0;}).forEach(function(p){var w=Math.min(Math.abs(p.pnl_pct),100).toFixed(0);html+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px"><span style="width:52px;color:var(--cyn);font-weight:700">'+p.ticker+'</span><div style="flex:1;height:5px;background:var(--bdr)"><div style="width:'+w+'%;height:100%;background:var(--dn)"></div></div><span style="color:var(--dn);width:60px;text-align:right">'+p.pnl_pct.toFixed(2)+'%</span></div>';});
    html+='</div></div>';
  }
  if(_rptSec('rpt-risk')){
    html+=_rptH('RISK METRICS');
    var pm2=_deepData&&_deepData.portfolio_metrics;
    html+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px"><div>';
    html+=_rptKV('HHI',s.hhi.toFixed(0)+' / 10000',s.concentration==='HIGH'?'var(--dn)':'var(--up)');
    html+=_rptKV('Concentration',s.concentration,s.concentration==='HIGH'?'var(--dn)':'var(--up)');
    if(_deepData){html+=_rptKV('VaR 95%','-'+(_deepData.var95_pct||0).toFixed(2)+'%','var(--dn)');html+=_rptKV('CVaR 95%','-'+(_deepData.cvar95_pct||0).toFixed(2)+'%','var(--dn)');}
    var bigPos=d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;})[0];
    if(bigPos)html+=_rptKV('Largest Position',bigPos.ticker+' ('+bigPos.weight_pct.toFixed(1)+'%)');
    html+='</div><div>';
    if(pm2){html+=_rptKV('Max Drawdown','-'+pm2.max_drawdown.toFixed(2)+'%','var(--dn)');html+=_rptKV('Portfolio Sharpe',pm2.sharpe.toFixed(2),pm2.sharpe>0?'var(--up)':'var(--dn)');html+=_rptKV('Portfolio Sortino',pm2.sortino.toFixed(2));html+=_rptKV('Calmar',pm2.calmar.toFixed(2));}
    html+=_rptKV('Cash Buffer',s.cash_pct.toFixed(1)+'%');
    html+='</div></div>';
  }
  if(_rptSec('rpt-allocation')){
    html+=_rptH('ALLOCATION BREAKDOWN');
    html+=_rptTable(['POSITION','MARKET VALUE','ALLOCATION','COST BASIS','UNREALISED P&L'],
      d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;}).map(function(p){
        var bar='<div style="display:flex;align-items:center;gap:4px"><div style="width:100px;height:4px;background:var(--bdr)"><div style="width:'+p.weight_pct.toFixed(1)+'%;height:100%;background:var(--org)"></div></div><span>'+p.weight_pct.toFixed(1)+'%</span></div>';
        return [{v:p.ticker,c:'var(--cyn)'},'$'+fk(p.market_value),bar,'$'+fk(p.cost),(p.pnl>=0?'+':'')+'$'+f(p.pnl,2)];
      }),['left','right','left','right','right']);
  }
  if(_rptSec('rpt-individual')){
    html+=_rptH('PER-POSITION ANALYTICS');
    html+=_rptTable(['TICKER','ANN VOL','SHARPE','SORTINO','MAX DD','WEIGHT','ENTRY DATE'],
      d.positions.map(function(p){
        return [{v:p.ticker,c:'var(--cyn)'},
          {v:p.ann_vol!=null?p.ann_vol.toFixed(1)+'%':'—',c:p.ann_vol>80?'var(--dn)':p.ann_vol>40?'var(--yel)':'var(--up)'},
          {v:p.sharpe!=null?p.sharpe.toFixed(2):'—',c:p.sharpe>1?'var(--up)':p.sharpe>0?'var(--yel)':'var(--dn)'},
          {v:p.sortino!=null?p.sortino.toFixed(2):'—'},
          {v:p.max_dd!=null?p.max_dd.toFixed(2)+'%':'—',c:'var(--dn)'},
          p.weight_pct.toFixed(1)+'%',p.entry_date||'—'];
      }),['left','right','right','right','right','right','left']);
  }
  if(_rptSec('rpt-monthly')&&_deepData&&_deepData.monthly_returns.length){
    html+=_rptH('MONTHLY RETURNS');
    html+=_rptTable(['MONTH','RETURN%'],
      _deepData.monthly_returns.map(function(m){return [m.month,{v:(m.return_pct>=0?'+':'')+m.return_pct.toFixed(2)+'%',c:m.return_pct>=0?'var(--up)':'var(--dn)'}];}),
      ['left','right']);
  }
  if(_rptSec('rpt-attribution')&&_deepData&&_deepData.attribution.length){
    html+=_rptH('RETURN ATTRIBUTION');
    html+=_rptTable(['TICKER','RETURN%','WEIGHT%','CONTRIBUTION%','ANN VOL','SHARPE'],
      _deepData.attribution.map(function(a){
        return [{v:a.ticker,c:'var(--cyn)'},{v:(a.return_pct>=0?'+':'')+a.return_pct.toFixed(2)+'%',c:a.return_pct>=0?'var(--up)':'var(--dn)'},a.weight_pct.toFixed(1)+'%',{v:(a.contribution_pct>=0?'+':'')+a.contribution_pct.toFixed(3)+'%',c:a.contribution_pct>=0?'var(--up)':'var(--dn)'},a.ann_vol!=null?a.ann_vol.toFixed(1)+'%':'—',a.sharpe!=null?a.sharpe.toFixed(2):'—'];
      }),['left','right','right','right','right','right']);
  }
  if(_rptSec('rpt-stress')&&_deepData&&_deepData.stress_tests){
    html+=_rptH('STRESS TEST RESULTS');
    html+=_rptTable(['SCENARIO','PORTFOLIO IMPACT%','EST. LOSS ($)'],
      _deepData.stress_tests.map(function(s){
        var aum=d.summary.total_with_cash;var lossVal=aum*s.portfolio_impact_pct/100;
        return [s.name,{v:(s.portfolio_impact_pct>=0?'+':'')+s.portfolio_impact_pct.toFixed(2)+'%',c:s.portfolio_impact_pct>=0?'var(--up)':'var(--dn)'},{v:(lossVal>=0?'+':'')+'$'+f(lossVal,2),c:lossVal>=0?'var(--up)':'var(--dn)'}];
      }),['left','right','right']);
  }
  if(_rptSec('rpt-realized')){
    var trades3=[];
    (_pfData.positions||[]).forEach(function(p){(p.closes||[]).forEach(function(c){trades3.push({ticker:p.ticker,close_price:c.close_price,close_qty:c.close_qty,close_date:c.close_date,realised_pnl:c.realised_pnl,notes:c.notes});});});
    if(trades3.length){
      html+=_rptH('REALISED P&L — CLOSED TRADES');
      html+=_rptTable(['TICKER','DATE','QTY','CLOSE PRICE','REALISED P&L','NOTES'],
        trades3.map(function(t){return [{v:t.ticker||'?',c:'var(--cyn)'},t.close_date||'—',t.close_qty||'—','$'+f(t.close_price||0,4),{v:(t.realised_pnl>=0?'+':'')+'$'+f(t.realised_pnl||0,2),c:t.realised_pnl>=0?'var(--up)':'var(--dn)'},t.notes||''];}),
        ['left','left','right','right','right','left']);
    }
  }
  if(_rptSec('rpt-cashlog')&&d.cash_log&&d.cash_log.length){
    html+=_rptH('CASH ACTIVITY LOG');
    var filteredLog=_rptDateFilter(d.cash_log,'ts').slice().reverse();
    html+=_rptTable(['DATE','AMOUNT','BALANCE AFTER','NOTE'],
      filteredLog.map(function(e){
        return [new Date(e.ts).toLocaleDateString('en-GB'),{v:(e.amount>=0?'+':'')+'$'+f(e.amount,2),c:e.amount>=0?'var(--up)':'var(--dn)'},'$'+f(e.balance_after||0,2),e.note||'—'];
      }),['left','right','right','left']);
  }
  if(_rptSec('rpt-notes')&&d.notes){html+=_rptH('PORTFOLIO NOTES');html+='<div style="background:var(--bg3);padding:12px;color:var(--txt2);line-height:1.7">'+d.notes+'</div>';}
  if(_rptSec('rpt-disclaimer')){
    html+='<div style="margin-top:32px;padding-top:16px;border-top:1px solid var(--bdr);font-size:9px;color:var(--txt3);line-height:1.6">'
      +'<div style="font-weight:700;margin-bottom:4px;letter-spacing:1px">DISCLAIMER</div>'
      +'This report is provided for informational purposes only and does not constitute investment advice. Past performance is not indicative of future results. '
      +'All figures are based on data available at time of generation. VaR, Sharpe Ratio, and Volatility are computed from historical price data. '
      +'Metrics marked "—" indicate insufficient price history. Generated by '+firm+'.'
      +'</div>';
  }
  html+='<div style="margin-top:24px;padding-top:8px;border-top:1px solid var(--bdr);font-size:9px;color:var(--txt3);text-align:right">'+firm+' · '+date+'</div>';
  html+='</div>';
  el.innerHTML=html;
};

window.printReport = function(){
  var content=document.getElementById('epf-report-body');
  if(!content){alert('Generate the report first.');return;}
  var w=window.open('','_blank');
  w.document.write('<html><head><title>Portfolio Report</title><style>body{background:#111;color:#bbb;font-family:"Courier New",monospace;padding:30px;font-size:11px;line-height:1.7}table{width:100%;border-collapse:collapse}th,td{padding:4px 6px}@media print{body{background:#fff;color:#000}}</style></head><body>'+content.innerHTML+'</body></html>');
  w.document.close();w.print();
};

window.exportReportCSV = function(){
  if(!_pfData){alert('No data.');return;}
  var d=_pfData;
  var lines=['PORTFOLIO REPORT,'+d.name+','+new Date().toLocaleDateString('en-GB'),'',
    'CLIENT,'+d.client,'PORTFOLIO,'+d.name,'STRATEGY,'+(d.strategy||''),'MARKET VALUE,$'+d.summary.total_value,'CASH,$'+d.cash,'TOTAL AUM,$'+d.summary.total_with_cash,'UNREALISED PNL,$'+d.summary.total_pnl,'RETURN %,'+d.summary.total_pnl_pct,'HHI,'+d.summary.hhi,'','HOLDINGS','TICKER,TYPE,QTY,ENTRY,LIVE,MKT VALUE,PNL,PNL%,WEIGHT%,ANN VOL%,SHARPE,SORTINO,MAX DD,ENTRY DATE'];
  d.positions.forEach(function(p){lines.push([p.ticker,(p.type||'long'),p.qty,p.entry_price,p.live_price||'',p.market_value,p.pnl,p.pnl_pct,p.weight_pct,p.ann_vol||'',p.sharpe||'',p.sortino||'',p.max_dd||'',p.entry_date||''].join(','));});
  if(d.cash_log&&d.cash_log.length){lines.push('','CASH LOG','DATE,AMOUNT,BALANCE AFTER,NOTE');d.cash_log.forEach(function(e){lines.push([new Date(e.ts).toLocaleDateString('en-GB'),e.amount,e.balance_after||0,(e.note||'').replace(/,/g,';')].join(','));});}
  var blob=new Blob([lines.join('\n')],{type:'text/csv'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=(d.name||'portfolio').replace(/\s+/g,'_')+'_report.csv';a.click();
};

window.exportStressExcel = function(){
  if(typeof XLSX === 'undefined'){alert('SheetJS not loaded yet, try again.');return;}
  if(!_deepData||!_deepData.stress_tests){alert('Load Analytics tab first to compute stress tests.');return;}
  var pfName=(_pfData&&_pfData.name)||'Portfolio';
  var aum=(_pfData&&_pfData.summary&&_pfData.summary.total_with_cash)||0;
  var rows=[['Scenario','Portfolio Impact %','Est. Loss ($)','AUM ($)']];
  _deepData.stress_tests.forEach(function(s){
    var loss=aum*s.portfolio_impact_pct/100;
    rows.push([s.name,s.portfolio_impact_pct.toFixed(2),loss.toFixed(2),aum.toFixed(2)]);
  });
  var ws=XLSX.utils.aoa_to_sheet(rows);
  ws['!cols']=[{wch:25},{wch:20},{wch:20},{wch:15}];
  var wb=XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb,ws,'Stress Tests');
  XLSX.writeFile(wb,pfName.replace(/\s+/g,'_')+'_stress_tests.xlsx');
};

// Report date range: filter cash_log and positions by selected dates
function _rptDateFilter(entries, tsField){
  tsField = tsField||'ts';
  var fromEl=document.getElementById('rpt-date-from');
  var toEl=document.getElementById('rpt-date-to');
  var from=fromEl&&fromEl.value?fromEl.value:'';
  var to=toEl&&toEl.value?toEl.value:new Date().toISOString().slice(0,10);
  return entries.filter(function(e){
    var d=(e[tsField]||'').slice(0,10);
    if(!d) return true;
    return (!from||d>=from) && (!to||d<=to);
  });
}
window._rptDateFilter = _rptDateFilter;



// ── NEW FEATURES BLOCK ──────────────────────────────────────────────────────

// ── Extend epfTab for new tabs (inside IIFE, has access to private vars) ────
var _origEpfTab = window.epfTab;
window.epfTab = function(tab){
  _origEpfTab(tab);
  if(tab==='drawdown')  loadDrawdown();
  if(tab==='beta')      loadRollingBeta();
  if(tab==='varbt')     loadVarBacktest();
  if(tab==='dividends') loadDividends();
  if(tab==='scenarios') loadScenarios();
  if(tab==='snapshots') loadSnapshots();
  if(tab==='tearsheet') { if(_pfData) generateTearsheet(); }
};

window.addPosition = async function(){
  var type = document.getElementById('apm-type').value;
  if(type === 'close'){
    var posId      = document.getElementById('close-pos-select').value;
    var closePrice = parseFloat(document.getElementById('close-price').value);
    var closeQty   = parseFloat(document.getElementById('close-qty').value)||null;
    var closeDate  = document.getElementById('close-date').value;
    var notes      = document.getElementById('apm-notes').value;
    if(!posId){alert('Select a position to close.');return;}
    if(!closePrice||closePrice<=0){alert('Enter a valid close price.');return;}
    var pos=_pfData&&_pfData.positions.find(function(p){return p.id===posId;});
    var ticker=pos?pos.ticker:'?';
    var body={close_price:closePrice,close_date:closeDate,notes:notes};
    if(closeQty)body.close_qty=closeQty;
    var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId+'/close',body);
    if(!r.ok){alert('Error closing: '+(r.d.detail||'unknown'));return;}
    hideAddPosModal();
    document.getElementById('apm-type').value='long'; toggleCloseMode('long');
    document.getElementById('apm-notes').value='';
    await ENT_PF_refresh(); epfTab('realized');
    var pnl=r.d.realised_pnl;
    var bar=document.getElementById('cmd-st');
    if(bar){bar.textContent=(r.d.removed?'CLOSED ':'PARTIAL CLOSE ')+ticker+' | PnL: '+(pnl>=0?'+':'')+'$'+f(pnl,2);bar.style.color=pnl>=0?'var(--up)':'var(--dn)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},4000);}
    return;
  }
  var ticker=(document.getElementById('apm-ticker').value||'').trim().toUpperCase();
  var qty=parseFloat(document.getElementById('apm-qty').value)||0;
  var price=parseFloat(document.getElementById('apm-price').value)||0;
  var date=document.getElementById('apm-date').value;
  var notes=document.getElementById('apm-notes').value;
  var tag=(document.getElementById('apm-tag')||{}).value||'';
  var stop=parseFloat((document.getElementById('apm-stop')||{}).value)||null;
  if(!ticker||!qty||!price){alert('Ticker, quantity and price required.');return;}
  var body={ticker:ticker,qty:qty,entry_price:price,entry_date:date,notes:notes,type:type};
  if(tag)  body.tag=tag;
  if(stop) body.stop_price=stop;
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions',body);
  if(!r.ok){alert('Error adding position');return;}
  hideAddPosModal();
  ['apm-ticker','apm-qty','apm-price','apm-notes','apm-tag','apm-stop'].forEach(function(id){var el=document.getElementById(id);if(el)el.value='';});
  await ENT_PF_refresh(); epfTab('positions');
};


// ── Position equity curve modal ────────────────────────────────────────────
window.showPositionCurve = async function(posId, ticker){
  var r = await api('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId+'/curve');
  if(!r.ok||!r.d.closes.length){alert('No price history for '+ticker);return;}
  var d=r.d;
  // Build modal on-the-fly
  var existing=document.getElementById('pos-curve-modal');
  if(existing) existing.remove();
  var modal=document.createElement('div');
  modal.id='pos-curve-modal';
  modal.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:3000;display:flex;align-items:center;justify-content:center';
  modal.innerHTML='<div style="background:var(--bg2);border:1px solid var(--org);width:640px;max-height:80vh;overflow:hidden;display:flex;flex-direction:column">'
    +'<div style="padding:10px 14px;background:var(--bg3);border-bottom:1px solid var(--bdr);font-size:10px;font-weight:700;letter-spacing:2px;color:var(--org);display:flex;justify-content:space-between">'
    +ticker+' — PRICE CHART SINCE ENTRY'
    +'<span style="cursor:pointer;color:var(--txt2)" onclick="document.getElementById(\'pos-curve-modal\').remove()">✕</span></div>'
    +'<div style="padding:10px;display:flex;gap:16px;font-size:10px;flex-shrink:0">'
    +'<span style="color:var(--txt3)">Entry: <b style="color:var(--wht)">$'+f(d.entry,4)+'</b></span>'
    +'<span style="color:var(--txt3)">Current: <b style="color:var(--wht)">'+(d.current?'$'+f(d.current,4):'—')+'</b></span>'
    +'<span style="color:var(--txt3)">P&L: <b style="color:'+(d.pnl_series.length&&d.pnl_series[d.pnl_series.length-1]>=0?'var(--up)':'var(--dn)')+'">'+( d.pnl_series.length?(d.pnl_series[d.pnl_series.length-1]>=0?'+':'')+d.pnl_series[d.pnl_series.length-1].toFixed(2)+'%':'—')+'</b></span>'
    +(d.stop_price?'<span style="color:var(--dn)">Stop: $'+f(d.stop_price,4)+'</span>':'')
    +'</div>'
    +'<div style="height:280px;padding:0 10px 10px;flex:1"><canvas id="pos-curve-canvas"></canvas></div>'
    +'</div>';
  document.body.appendChild(modal);
  var ctx=document.getElementById('pos-curve-canvas');
  var up=d.closes[d.closes.length-1]>=d.entry;
  var stopLine=d.stop_price&&d.stop_price>0?[{type:'line',yMin:d.stop_price,yMax:d.stop_price,borderColor:'rgba(244,67,54,.6)',borderWidth:1,borderDash:[4,2]}]:[];
  new Chart(ctx,{
    type:'line',
    data:{labels:d.dates,datasets:[
      {data:d.closes,borderColor:up?'#00c853':'#f44336',borderWidth:1.8,pointRadius:0,pointHoverRadius:4,fill:true,backgroundColor:up?'rgba(0,200,83,.07)':'rgba(244,67,54,.06)'},
      {data:Array(d.closes.length).fill(d.entry),borderColor:'rgba(255,140,0,.4)',borderWidth:1,borderDash:[5,3],pointRadius:0,fill:false,label:'Entry'}
    ]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return c.datasetIndex===0?' $'+f(c.parsed.y,4):' Entry $'+f(c.parsed.y,4);}}}},
      scales:{x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#444',maxTicksLimit:8,font:{size:8}}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',font:{size:9},callback:function(v){return '$'+f(v,2);}}}}}
  });
  modal.onclick=function(e){if(e.target===modal)modal.remove();};
};

// ── Drawdown chart ────────────────────────────────────────────────────────
window.loadDrawdown = async function(){
  if(!_deepData) await loadDeepAnalytics();
  if(!_deepData||!_deepData.port_closes||!_deepData.port_closes.length){
    var el=document.getElementById('epf-dd-stats');
    if(el) el.innerHTML='<div style="padding:12px;color:var(--yel);font-size:10px;grid-column:1/-1">Open ANALYTICS tab first to load price history.</div>';
    return;
  }
  var pts=_deepData.port_closes;
  // Compute rolling drawdown series
  var peak=pts[0]; var ddSeries=[];
  for(var i=0;i<pts.length;i++){
    if(pts[i]>peak) peak=pts[i];
    ddSeries.push(peak>0?-(peak-pts[i])/peak*100:0);
  }
  // Drawdown chart
  var ctx=document.getElementById('epf-drawdown-chart');
  if(ctx){
    if(_charts.drawdown) _charts.drawdown.destroy();
    var labels=pts.map(function(_,i){return i;});
    _charts.drawdown=new Chart(ctx,{
      type:'line',
      data:{labels:labels,datasets:[{data:ddSeries,borderColor:'#f44336',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(244,67,54,.12)',tension:0.1}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return ' Drawdown: '+c.parsed.y.toFixed(2)+'%';}}}},
        scales:{x:{display:false},y:{grid:{color:'#1a1a1a'},position:'right',max:0,ticks:{color:'#444',callback:function(v){return v.toFixed(0)+'%';}}}}}});
  }
  // Equity + drawdown overlay (dual axis)
  var ctx2=document.getElementById('epf-dd-equity-chart');
  if(ctx2){
    if(_charts.ddEquity) _charts.ddEquity.destroy();
    var labels2=pts.map(function(_,i){return i;});
    _charts.ddEquity=new Chart(ctx2,{
      type:'line',
      data:{labels:labels2,datasets:[
        {data:pts,borderColor:'#00c853',borderWidth:1.5,pointRadius:0,fill:false,yAxisID:'y1'},
        {data:ddSeries,borderColor:'rgba(244,67,54,.5)',borderWidth:1,pointRadius:0,fill:true,backgroundColor:'rgba(244,67,54,.07)',yAxisID:'y2'}
      ]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{display:false},tooltip:{...TT}},
        scales:{
          x:{display:false},
          y1:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',font:{size:9},callback:function(v){return '$'+fk(v);}}},
          y2:{grid:{display:false},position:'left',max:0,ticks:{color:'#666',font:{size:8},callback:function(v){return v.toFixed(0)+'%';}}}
        }}});
  }
  // Stats
  var maxDD=Math.min.apply(null,ddSeries);
  var maxDDIdx=ddSeries.indexOf(maxDD);
  // Find longest underwater period
  var inDD=false; var curLen=0; var maxLen=0;
  ddSeries.forEach(function(d){if(d<0){inDD=true;curLen++;maxLen=Math.max(maxLen,curLen);}else{inDD=false;curLen=0;}});
  var statsEl=document.getElementById('epf-dd-stats');
  if(statsEl){
    var items=[
      {l:'MAX DRAWDOWN',v:maxDD.toFixed(2)+'%',c:'var(--dn)'},
      {l:'MAX DD AT DAY',v:'D'+maxDDIdx,c:'var(--txt2)'},
      {l:'CURRENT DD',v:ddSeries[ddSeries.length-1].toFixed(2)+'%',c:ddSeries[ddSeries.length-1]<-5?'var(--dn)':'var(--yel)'},
      {l:'LONGEST UNDERWATER',v:maxLen+'d',c:'var(--yel)'},
    ];
    statsEl.innerHTML=items.map(function(it){return '<div style="background:var(--bg2);padding:10px 14px"><div style="font-size:8px;color:var(--txt3);letter-spacing:1.5px;margin-bottom:4px">'+it.l+'</div><div style="font-size:16px;font-weight:700;color:'+it.c+'">'+it.v+'</div></div>';}).join('');
  }
}

// ── Rolling beta ──────────────────────────────────────────────────────────
window.loadRollingBeta = async function(){
  if(!_pfid) return;
  var days=(document.getElementById('beta-days')||{}).value||'90';
  var r=await api('/api/enterprise/portfolios/'+_pfid+'/rolling_beta?days='+days);
  if(!r.ok) return;
  var d=r.d;
  var ctx=document.getElementById('epf-beta-chart');
  if(ctx){
    if(_charts.beta) _charts.beta.destroy();
    var n20=d.beta20.length; var n60=d.beta60.length;
    var labels=Array.from({length:Math.max(n20,n60)},function(_,i){return i;});
    var ds=[{label:'20D Beta',data:d.beta20,borderColor:'#ff8c00',borderWidth:1.8,pointRadius:0,fill:false}];
    if(n60>0) ds.push({label:'60D Beta',data:d.beta60,borderColor:'#534AB7',borderWidth:1.5,borderDash:[4,2],pointRadius:0,fill:false});
    // Neutral line at 1
    ds.push({data:Array(Math.max(n20,n60)).fill(1),borderColor:'rgba(255,255,255,.1)',borderWidth:1,borderDash:[2,4],pointRadius:0,fill:false,label:'β=1'});
    _charts.beta=new Chart(ctx,{
      type:'line',
      data:{labels:labels,datasets:ds},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        interaction:{mode:'index',intersect:false},
        plugins:{legend:{labels:{color:'#888',font:{size:9},boxWidth:10}},tooltip:{...TT,callbacks:{label:function(c){return ' '+c.dataset.label+': '+c.parsed.y.toFixed(3);}}}},
        scales:{x:{display:false},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',callback:function(v){return v.toFixed(2);}}}}}});
  }
  var detEl=document.getElementById('epf-beta-detail');
  if(detEl&&d.beta20.length){
    var latest20=d.beta20[d.beta20.length-1];
    var latest60=d.beta60.length?d.beta60[d.beta60.length-1]:null;
    var interp=latest20>1.2?'High market sensitivity — amplifies market moves.':latest20>0.8?'Market-neutral to slightly correlated.':latest20>0.3?'Defensive — moves less than the market.':'Very low or negative correlation — potential hedge.';
    detEl.innerHTML='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:10px">'
      +'<div><div style="font-size:8px;color:var(--txt3);margin-bottom:4px">CURRENT 20D BETA</div><div style="font-size:22px;font-weight:700;color:var(--yel)">'+latest20.toFixed(3)+'</div></div>'
      +(latest60!==null?'<div><div style="font-size:8px;color:var(--txt3);margin-bottom:4px">CURRENT 60D BETA</div><div style="font-size:22px;font-weight:700;color:var(--org)">'+latest60.toFixed(3)+'</div></div>':'')
      +'<div style="grid-column:1/-1;color:var(--txt2);font-style:italic">'+interp+'</div>'
      +'<div style="grid-column:1/-1;color:var(--txt3)">Observations: '+d.n_obs+'  |  β&gt;1 = amplified market exposure  |  β&lt;0 = inverse</div>'
      +'</div>';
  }
}

// ── VaR Backtest ──────────────────────────────────────────────────────────
window.loadVarBacktest = async function(){
  if(!_pfid) return;
  var r=await api('/api/enterprise/portfolios/'+_pfid+'/var_backtest?days=180');
  if(!r.ok||!r.d.n_test) return;
  var d=r.d;
  var kpis=document.getElementById('epf-varbt-kpis');
  if(kpis){
    var c95=d.coverage95!=null?d.coverage95*100:null;
    var c99=d.coverage99!=null?d.coverage99*100:null;
    var status95=d.breaches95<=Math.ceil(d.expected95*2)?'PASS':'FAIL';
    kpis.innerHTML=[
      {l:'VAR 95%',v:'-'+d.var95_pct.toFixed(3)+'%',c:'var(--dn)'},
      {l:'BREACHES 95% (expected '+d.expected95+')',v:d.breaches95+' / '+d.n_test,c:status95==='PASS'?'var(--up)':'var(--dn)'},
      {l:'COVERAGE 95%',v:c95!=null?c95.toFixed(1)+'%':'—',c:c95>=93?'var(--up)':'var(--dn)'},
      {l:'STATUS',v:status95,c:status95==='PASS'?'var(--up)':'var(--dn)'},
    ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');
  }
  // Chart: daily returns with VaR line + breach markers
  var ctx=document.getElementById('epf-varbt-chart');
  if(ctx){
    if(_charts.varbt) _charts.varbt.destroy();
    var rets=d.port_rets;
    var halfN=Math.floor(rets.length/2);
    var varLine=Array(halfN).fill(null).concat(Array(rets.length-halfN).fill(-d.var95_pct));
    _charts.varbt=new Chart(ctx,{
      type:'bar',
      data:{
        labels:rets.map(function(_,i){return i;}),
        datasets:[
          {data:rets,backgroundColor:rets.map(function(v){return v>=0?'rgba(0,200,83,.5)':'rgba(244,67,54,.5)';}),borderWidth:0,label:'Daily Return %'},
          {type:'line',data:varLine,borderColor:'rgba(255,140,0,.8)',borderWidth:1.5,borderDash:[4,2],pointRadius:0,fill:false,label:'VaR 95%'}
        ]
      },
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{labels:{color:'#888',font:{size:9}}},tooltip:{...TT,callbacks:{label:function(c){return ' '+c.dataset.label+': '+(c.parsed.y||0).toFixed(3)+'%';}}}},
        scales:{x:{display:false},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#444',callback:function(v){return v.toFixed(2)+'%';}}}}}});
  }
  var detEl=document.getElementById('epf-varbt-detail');
  if(detEl){
    var ok=d.breaches95<=Math.ceil(d.expected95*2);
    detEl.innerHTML='<div style="font-size:10px;color:var(--txt2);line-height:1.8">'
      +'VaR 95% estimated at <b style="color:var(--dn)">-'+d.var95_pct.toFixed(3)+'%</b> from the first '+Math.floor(d.port_rets.length/2)+' days.<br>'
      +'In the test period ('+d.n_test+' days), the actual loss exceeded the VaR estimate <b style="color:'+(ok?'var(--up)':'var(--dn)')+'">'+d.breaches95+'</b> times (expected ~'+d.expected95+').<br>'
      +'Coverage ratio: <b>'+((d.coverage95||0)*100).toFixed(1)+'%</b> (target ≥ 95%).<br>'
      +'<span style="color:'+(ok?'var(--up)':'var(--dn)')+';font-weight:700">'+( ok?'✓ Model appears well-calibrated.':'⚠ Model may be underestimating risk — more breaches than expected.')+'</span>'
      +'</div>';
  }
}

// ── Dividends ─────────────────────────────────────────────────────────────
window.loadDividends = async function(){
  if(!_pfData) return;
  var el=document.getElementById('epf-dividends-body');
  if(!el) return;
  // Collect dividends from all positions
  var rows=[];
  var totalDiv=0;
  var divSrc=(_pfData.all_dividends&&_pfData.all_dividends.length)?_pfData.all_dividends:(function(){var o=[];_pfData.positions.forEach(function(p){(p.dividends||[]).forEach(function(d){o.push(Object.assign({ticker:p.ticker},d));});});return o;})();
  divSrc.forEach(function(d){rows.push({ticker:d.ticker,date:d.date,amount:d.amount,note:d.note,ts:d.ts});totalDiv+=parseFloat(d.amount)||0;});
  rows.sort(function(a,b){return (b.date||b.ts||'').localeCompare(a.date||a.ts||'');});
  if(!rows.length){
    el.innerHTML='<div style="color:var(--txt3);font-size:10px;padding:10px">No dividends recorded yet. Use + RECORD DIVIDEND to log income.</div>';
    return;
  }
  el.innerHTML='<div style="font-size:10px;color:var(--up);font-weight:700;margin-bottom:10px">Total Income: +$'+f(totalDiv,2)+'</div>'
    +'<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th>DATE</th><th class="r">AMOUNT</th><th>NOTE</th></tr></thead><tbody>'
    +rows.map(function(r){
      return '<tr><td style="color:var(--cyn);font-weight:700">'+r.ticker+'</td>'
        +'<td>'+(r.date||'—')+'</td>'
        +'<td class="r" style="color:var(--up)">+$'+f(r.amount,2)+'</td>'
        +'<td>'+(r.note||'')+'</td></tr>';
    }).join('')+'</tbody></table>';
}

window.showDividendModal = function(){
  var m=document.getElementById('dividend-modal');
  if(!m) return;
  if(m.parentElement!==document.body) document.body.appendChild(m);
  var sel=document.getElementById('div-pos-select');
  if(sel&&_pfData){
    sel.innerHTML='<option value="">— select —</option>'+_pfData.positions.map(function(p){return '<option value="'+p.id+'">'+p.ticker+'</option>';}).join('');
  }
  var dt=document.getElementById('div-date');
  if(dt&&!dt.value) dt.value=new Date().toISOString().slice(0,10);
  m.style.display='flex';
};
window.hideDividendModal=function(){var m=document.getElementById('dividend-modal');if(m)m.style.display='none';};
window.recordDividend = async function(){
  var posId=document.getElementById('div-pos-select').value;
  var amount=parseFloat(document.getElementById('div-amount').value)||0;
  var date=document.getElementById('div-date').value;
  var note=document.getElementById('div-note').value;
  if(!posId||!amount){alert('Select position and enter amount.');return;}
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/positions/'+posId+'/dividends',{amount:amount,date:date,note:note});
  if(!r.ok){alert('Error recording dividend');return;}
  hideDividendModal();
  ['div-amount','div-note'].forEach(function(id){var el=document.getElementById(id);if(el)el.value='';});
  await ENT_PF_refresh(); epfTab('dividends');
};

// ── Scenario builder ──────────────────────────────────────────────────────
window.loadScenarios = async function(){
  if(!_pfid) return;
  var r=await api('/api/enterprise/portfolios/'+_pfid+'/scenarios');
  if(!r.ok) return;
  var el=document.getElementById('epf-scenarios-list');
  if(!el) return;
  var saved=r.d;
  if(!saved.length){el.innerHTML='<div style="color:var(--txt3);font-size:10px;padding:10px">No saved scenarios yet.</div>';return;}
  el.innerHTML=saved.map(function(sc){
    var col=sc.total_impact_pct>=0?'var(--up)':'var(--dn)';
    return '<div style="border:1px solid var(--bdr);padding:10px;margin-bottom:8px;font-size:10px">'
      +'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
      +'<b style="color:var(--wht)">'+sc.name+'</b>'
      +'<div style="display:flex;gap:8px;align-items:center">'
      +'<span style="font-size:14px;font-weight:700;color:'+col+'">'+(sc.total_impact_pct>=0?'+':'')+sc.total_impact_pct.toFixed(2)+'%</span>'
      +'<span style="color:'+col+'">'+(sc.est_loss>=0?'+':'')+'$'+f(Math.abs(sc.est_loss),2)+'</span>'
      +'<span style="cursor:pointer;color:var(--dn)" onclick="deleteScenario(\''+sc.id+'\')">✕</span>'
      +'</div></div>'
      +'<div style="color:var(--txt3)">'+sc.impacts.map(function(i){return i.ticker+' '+( i.shock_pct>=0?'+':'')+i.shock_pct.toFixed(0)+'%';}).join(' · ')+'</div>'
      +'</div>';
  }).join('');
}

window.addScenarioRow = function(){
  var rows=document.getElementById('sc-rows');
  if(!rows||!_pfData) return;
  var opts=_pfData.positions.map(function(p){return '<option value="'+p.ticker+'">'+p.ticker+'</option>';}).join('');
  var div=document.createElement('div');
  div.style.cssText='display:flex;gap:8px;align-items:center';
  div.innerHTML='<select style="background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:3px 6px;outline:none">'+opts+'</select>'
    +'<input type="number" placeholder="shock%" style="width:80px;background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:3px 6px;outline:none">'
    +'<span style="cursor:pointer;color:var(--dn)" onclick="this.parentElement.remove()">✕</span>';
  rows.appendChild(div);
};

window.runScenario = async function(save){
  var name=(document.getElementById('sc-name')||{}).value||'Custom Scenario';
  var rows=document.querySelectorAll('#sc-rows > div');
  var shocks={};
  rows.forEach(function(row){
    var sel=row.querySelector('select');
    var inp=row.querySelector('input');
    if(sel&&inp&&inp.value) shocks[sel.value]=parseFloat(inp.value)/100;
  });
  if(!Object.keys(shocks).length){alert('Add at least one ticker shock.');return;}
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/scenarios',{name:name,shocks:shocks,save:save});
  if(!r.ok){alert('Error running scenario');return;}
  var sc=r.d;
  var col=sc.total_impact_pct>=0?'var(--up)':'var(--dn)';
  var resEl=document.getElementById('sc-result');
  if(resEl){
    resEl.style.display='block';
    resEl.innerHTML='<div style="font-size:12px;font-weight:700;color:'+col+';margin-bottom:8px">'
      +(sc.total_impact_pct>=0?'+':'')+sc.total_impact_pct.toFixed(3)+'% portfolio impact'
      +' ('+(sc.est_loss>=0?'+':'')+'$'+f(Math.abs(sc.est_loss),2)+')</div>'
      +'<table class="dt" style="width:100%;font-size:9px"><thead><tr><th>TICKER</th><th class="r">WEIGHT</th><th class="r">SHOCK</th><th class="r">CONTRIBUTION</th></tr></thead><tbody>'
      +sc.impacts.map(function(i){var c=i.contribution_pct>=0?'var(--up)':'var(--dn)';return '<tr><td style="color:var(--cyn)">'+i.ticker+'</td><td class="r">'+i.weight_pct.toFixed(1)+'%</td><td class="r" style="color:'+c+'">'+(i.shock_pct>=0?'+':'')+i.shock_pct.toFixed(1)+'%</td><td class="r" style="font-weight:700;color:'+c+'">'+(i.contribution_pct>=0?'+':'')+i.contribution_pct.toFixed(3)+'%</td></tr>';}).join('')
      +'</tbody></table>'+(save?'<div style="color:var(--up);font-size:9px;margin-top:6px">✓ Scenario saved.</div>':'');
  }
  if(save) loadScenarios();
};

window.deleteScenario = async function(id){
  await api('/api/enterprise/portfolios/'+_pfid+'/scenarios?id='+id,{method:'DELETE'});
  loadScenarios();
};

// ── Snapshots ─────────────────────────────────────────────────────────────
window.loadSnapshots = async function(){
  if(!_pfid) return;
  var r=await api('/api/enterprise/portfolios/'+_pfid+'/report_snapshots');
  if(!r.ok) return;
  var el=document.getElementById('epf-snapshots-list');
  if(!el) return;
  var snaps=r.d;
  if(!snaps.length){el.innerHTML='<div style="color:var(--txt3);font-size:10px;padding:10px">No snapshots saved yet. Click SAVE CURRENT SNAPSHOT to capture the portfolio state.</div>';return;}
  el.innerHTML=snaps.slice().reverse().map(function(s){
    var sum=s.summary||{};
    return '<div style="border:1px solid var(--bdr);padding:12px;margin-bottom:8px">'
      +'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
      +'<div><span style="color:var(--org);font-weight:700;font-size:11px">'+s.label+'</span>'
      +'<span style="color:var(--txt3);font-size:9px;margin-left:10px">'+new Date(s.ts).toLocaleDateString('en-GB')+'</span></div>'
      +'<div style="display:flex;gap:8px">'
      +'<button class="btn btn-xs" onclick="viewSnapshot(\''+s.id+'\')">VIEW</button>'
      +'<button class="btn btn-xs" style="color:var(--dn);border-color:var(--dn)" onclick="deleteSnapshot(\''+s.id+'\')">DEL</button>'
      +'</div></div>'
      +(s.commentary?'<div style="font-size:10px;color:var(--txt2);font-style:italic;margin-bottom:6px">'+s.commentary+'</div>':'')
      +(sum.total_with_cash?'<div style="font-size:9px;color:var(--txt3)">AUM: $'+fk(sum.total_with_cash)+'  |  P&L: '+(sum.total_pnl>=0?'+':'')+'$'+f(sum.total_pnl,2)+'  |  Return: '+(sum.total_pnl_pct>=0?'+':'')+sum.total_pnl_pct.toFixed(2)+'%</div>':'')
      +'</div>';
  }).join('');
}

window.saveSnapshot = async function(){
  if(!_pfData){alert('No portfolio data loaded.');return;}
  var label=prompt('Snapshot label (e.g. "March 2026"):', new Date().toISOString().slice(0,7));
  if(!label) return;
  var tsBody=document.getElementById('epf-report-body');
  var html=tsBody?tsBody.innerHTML:'';
  var r=await apiPost('/api/enterprise/portfolios/'+_pfid+'/report_snapshots',{
    label:label, html:html, summary:_pfData.summary,
    commentary:(document.getElementById('ts-commentary')||{}).value||''
  });
  if(!r.ok){alert('Error saving snapshot');return;}
  var bar=document.getElementById('cmd-st');
  if(bar){bar.textContent='Snapshot saved: '+label;bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2500);}
  loadSnapshots();
};

window.viewSnapshot = async function(id){
  var r=await api('/api/enterprise/portfolios/'+_pfid+'/report_snapshots');
  if(!r.ok) return;
  var snap=r.d.find(function(s){return s.id===id;});
  if(!snap||!snap.html) return;
  var w=window.open('','_blank');
  w.document.write('<html><head><title>'+snap.label+'</title><style>body{background:#111;color:#bbb;font-family:"Courier New",monospace;padding:30px;font-size:11px;line-height:1.7}</style></head><body>'+snap.html+'</body></html>');
  w.document.close();
};

window.deleteSnapshot = async function(id){
  if(!confirm('Delete this snapshot?')) return;
  await api('/api/enterprise/portfolios/'+_pfid+'/report_snapshots?id='+id,{method:'DELETE'});
  loadSnapshots();
};

// ── Tearsheet ─────────────────────────────────────────────────────────────
window.generateTearsheet = function(){
  if(!_pfData) return;
  var d=_pfData; var s=d.summary;
  var pm=_deepData&&_deepData.portfolio_metrics;
  var bm=_bmData;
  var date=new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric'});
  var el=document.getElementById('epf-tearsheet-body');
  if(!el) return;
  var commentary=(document.getElementById('ts-commentary')||{}).value||'';
  var pnlC=s.total_pnl>=0?'#00c853':'#f44336';
  var html='<div style="font-family:\'Courier New\',monospace;color:#bbb;max-width:800px;margin:0 auto">'
    // Header
    +'<div style="display:flex;justify-content:space-between;align-items:flex-start;border-bottom:2px solid #ff8c00;padding-bottom:12px;margin-bottom:16px">'
    +'<div><div style="font-size:20px;font-weight:700;color:#fff;letter-spacing:1.5px">'+(d.name||'Portfolio')+'</div>'
    +'<div style="font-size:9px;color:#666;margin-top:2px">'+(d.client||'')+(d.strategy?' · '+d.strategy:'')+'</div></div>'
    +'<div style="text-align:right"><div style="font-size:9px;color:#666">FACTSHEET</div><div style="font-size:11px;color:#888">'+date+'</div></div>'
    +'</div>'
    // KPI strip
    +'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:#222;border:1px solid #222;margin-bottom:16px">';
  var kpis=[
    ['AUM','$'+fk(s.total_with_cash),'#fff'],
    ['RETURN',(s.total_pnl_pct>=0?'+':'')+s.total_pnl_pct.toFixed(2)+'%',pnlC],
    ['SHARPE',pm&&pm.sharpe!=null?pm.sharpe.toFixed(2):'—',pm&&pm.sharpe>0?'#00c853':'#f44336'],
    ['MAX DD',pm&&pm.max_drawdown!=null?'-'+pm.max_drawdown.toFixed(2)+'%':'—','#f44336'],
    ['VOL',pm&&pm.ann_vol!=null?pm.ann_vol.toFixed(1)+'%':'—','#ffd600'],
    ['POSITIONS',s.num_positions,'#fff'],
  ];
  kpis.forEach(function(k){html+='<div style="background:#111;padding:8px 10px;text-align:center"><div style="font-size:7px;color:#555;letter-spacing:1.5px;margin-bottom:3px">'+k[0]+'</div><div style="font-size:14px;font-weight:700;color:'+k[2]+'">'+k[1]+'</div></div>';});
  html+='</div>'
  // Two-column: holdings + allocation
    +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">'
    +'<div><div style="font-size:8px;color:#ff8c00;letter-spacing:2px;margin-bottom:6px">HOLDINGS</div>'
    +'<table style="width:100%;border-collapse:collapse;font-size:9px">'
    +'<tr style="border-bottom:1px solid #222"><th style="text-align:left;color:#666;padding:2px 4px">TICKER</th><th style="text-align:right;color:#666">VALUE</th><th style="text-align:right;color:#666">WEIGHT</th><th style="text-align:right;color:#666">P&L</th></tr>'
    +d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;}).map(function(p){
      return '<tr style="border-bottom:1px solid rgba(34,34,34,.5)"><td style="color:#00e5ff;padding:2px 4px;font-weight:700">'+p.ticker+'</td>'
        +'<td style="text-align:right">$'+fk(p.market_value)+'</td>'
        +'<td style="text-align:right">'+p.weight_pct.toFixed(1)+'%</td>'
        +'<td style="text-align:right;color:'+(p.pnl>=0?'#00c853':'#f44336')+'">'+(p.pnl>=0?'+':'')+'$'+f(p.pnl,2)+'</td></tr>';
    }).join('')+'</table></div>'
    // Allocation bars
    +'<div><div style="font-size:8px;color:#ff8c00;letter-spacing:2px;margin-bottom:6px">ALLOCATION</div>'
    +d.positions.slice().sort(function(a,b){return b.weight_pct-a.weight_pct;}).map(function(p){
      return '<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;font-size:9px">'
        +'<span style="width:44px;color:#00e5ff">'+p.ticker+'</span>'
        +'<div style="flex:1;height:5px;background:#1a1a1a"><div style="width:'+p.weight_pct.toFixed(1)+'%;height:100%;background:#ff8c00"></div></div>'
        +'<span style="width:36px;text-align:right">'+p.weight_pct.toFixed(1)+'%</span>'
        +'</div>';
    }).join('')+'</div></div>'
  // Benchmark row
    +(bm?'<div style="border:1px solid #222;padding:10px;margin-bottom:16px;font-size:9px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px">'
      +'<div><div style="color:#666;margin-bottom:2px">VS BENCHMARK</div><div style="color:#534AB7;font-weight:700">'+(bm.benchmark_label||'NER Index')+'</div></div>'
      +'<div><div style="color:#666;margin-bottom:2px">BM RETURN</div><div style="color:'+((bm.benchmark_return||0)>=0?'#00c853':'#f44336')+';font-weight:700">'+((bm.benchmark_return||0)>=0?'+':'')+(bm.benchmark_return!=null?bm.benchmark_return.toFixed(2):'—')+'%</div></div>'
      +'<div><div style="color:#666;margin-bottom:2px">ALPHA</div><div style="color:'+((bm.alpha||0)>=0?'#00c853':'#f44336')+';font-weight:700">'+((bm.alpha||0)>=0?'+':'')+(bm.alpha!=null?bm.alpha.toFixed(2):'—')+'%</div></div>'
      +'<div><div style="color:#666;margin-bottom:2px">BETA</div><div style="color:#ffd600;font-weight:700">'+(bm.beta!=null?bm.beta.toFixed(3):'—')+'</div></div>'
      +'</div>':'')
  // Monthly returns heatmap
    +(_deepData&&_deepData.monthly_returns.length?'<div style="margin-bottom:16px"><div style="font-size:8px;color:#ff8c00;letter-spacing:2px;margin-bottom:6px">MONTHLY RETURNS</div>'
      +'<div style="display:flex;flex-wrap:wrap;gap:3px">'
      +_deepData.monthly_returns.map(function(m){var i=Math.min(Math.abs(m.return_pct)/5,1);var bg=m.return_pct>=0?'rgba(0,200,83,'+i*0.7+')':'rgba(244,67,54,'+i*0.7+')';return '<div style="background:'+bg+';padding:3px 6px;min-width:68px;text-align:center"><div style="font-size:7px;color:rgba(255,255,255,.5)">'+m.month+'</div><div style="font-size:9px;font-weight:700;color:#fff">'+(m.return_pct>=0?'+':'')+m.return_pct.toFixed(2)+'%</div></div>';}).join('')
      +'</div></div>':'')
  // Commentary
    +(commentary?'<div style="border-left:3px solid #ff8c00;padding:8px 12px;margin-bottom:16px;font-size:10px;color:#999;font-style:italic">'+commentary+'</div>':'')
  // Footer
    +'<div style="border-top:1px solid #222;padding-top:8px;font-size:8px;color:#444;display:flex;justify-content:space-between">'
    +'<span>Generated by BLOOMBERG / NER ENTERPRISE</span><span>'+date+'</span>'
    +'</div></div>';
  el.innerHTML=html;
};

window.printTearsheet = function(){
  var c=document.getElementById('epf-tearsheet-body');
  if(!c){alert('Generate tearsheet first.');return;}
  var w=window.open('','_blank');
  w.document.write('<html><head><title>Tearsheet</title><style>body{background:#111;color:#bbb;font-family:"Courier New",monospace;padding:20px;font-size:11px}@media print{body{background:#fff;color:#000}}</style></head><body>'+c.innerHTML+'</body></html>');
  w.document.close(); w.print();
};

})();
