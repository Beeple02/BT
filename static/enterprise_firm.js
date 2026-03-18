// enterprise_firm.js — Firm Dashboard Space JS
(function(){

var _firmData = null;
var _firmCharts = {};
var _firmSettings = {};
var _realizedData = null;
var _attrData = null;
var _briefData = null;

var FIRM_TABS = ['brief','overview','exposure','rebalance','reviews','realized','attribution','model','compliance','revenue','settings'];

// Currency symbol helper — 10 R$ = 1 A£
function _ccy(){ return _firmSettings.currency==='AP' ? 'A£' : 'R$'; }
function _fmtAum(v){
  var sym = _ccy();
  if(_firmSettings.currency==='AP') v = v/10;  // convert from R$ base to A£
  return sym+fk(v);
}

window.FIRM_load = async function(){
  var r = await api('/api/enterprise/firm_settings');
  if(r.ok) _firmSettings = r.d;
  FIRM_applySettings();
  await Promise.all([FIRM_loadBrief(), FIRM_loadMain()]);
};

window.FIRM_tab = function(tab){
  document.querySelectorAll('.firm-tab').forEach(function(t,i){
    t.classList.toggle('on', FIRM_TABS[i]===tab);
  });
  document.querySelectorAll('.firm-panel').forEach(function(p){
    p.classList.toggle('on', p.id==='firm-'+tab);
  });
  setTimeout(function(){ Object.values(_firmCharts).forEach(function(c){ if(c&&c.resize) c.resize(); }); }, 60);
  if(tab==='realized'   && !_realizedData) FIRM_loadRealized();
  if(tab==='attribution') FIRM_loadAttribution();
  if(tab==='settings')  FIRM_loadSettings();
  if(tab==='reviews')   FIRM_renderReviews();
  if(tab==='rebalance') FIRM_renderRebalance();
};

// ── Settings ──────────────────────────────────────────────────────────────────
function FIRM_applySettings(){
  var s = _firmSettings;
  var set = function(id, v){ var el=document.getElementById(id); if(el&&v!=null) el.value=v; };
  set('s-firm-name',   s.firm_name);
  set('s-currency',    s.currency||'RD');
  set('s-disclaimer',  s.disclaimer);
  set('s-mgmt-fee',    s.mgmt_fee);
  set('s-perf-fee',    s.perf_fee);
  set('s-hurdle',      s.hurdle_rate);
  set('s-hwm',         s.hwm||'yes');
  set('s-max-pos',     s.max_position_pct||40);
  set('s-max-hhi',     s.max_hhi||4000);
  set('s-drift',       s.drift_alert_pct||5);
  set('s-idle-days',   s.idle_cash_days||30);
  set('s-review-freq', s.default_review_freq||'quarterly');
  // Update drift threshold display
  var th = document.getElementById('rebal-threshold');
  if(th) th.textContent = s.drift_alert_pct||5;
}

async function FIRM_loadSettings(){
  var r = await api('/api/enterprise/firm_settings');
  if(r.ok){ _firmSettings = r.d; FIRM_applySettings(); }
}

window.FIRM_saveSettings = async function(){
  var get = function(id){ var el=document.getElementById(id); return el?el.value:''; };
  var body = {
    firm_name:        get('s-firm-name'),
    currency:         get('s-currency'),
    disclaimer:       get('s-disclaimer'),
    mgmt_fee:         parseFloat(get('s-mgmt-fee'))||1,
    perf_fee:         parseFloat(get('s-perf-fee'))||20,
    hurdle_rate:      parseFloat(get('s-hurdle'))||5,
    hwm:              get('s-hwm'),
    max_position_pct: parseFloat(get('s-max-pos'))||40,
    max_hhi:          parseFloat(get('s-max-hhi'))||4000,
    drift_alert_pct:  parseFloat(get('s-drift'))||5,
    idle_cash_days:   parseFloat(get('s-idle-days'))||30,
    default_review_freq: get('s-review-freq'),
  };
  var r = await apiPost('/api/enterprise/firm_settings', body);
  var st = document.getElementById('settings-status');
  if(r.ok){
    _firmSettings = r.d;
    if(st){ st.textContent='✓ Settings saved — refreshing…'; st.style.color='var(--up)'; setTimeout(function(){st.textContent='';},3000); }
    var bar=document.getElementById('cmd-st');
    if(bar){bar.textContent='Firm settings saved';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);}
    // Re-render every panel that uses settings
    if(_firmData){
      FIRM_renderOverview();
      FIRM_renderRevenue();
      FIRM_renderCompliance();
    }
    if(_briefData) FIRM_renderBrief();
    var thEl=document.getElementById('rebal-threshold');
    if(thEl) thEl.textContent=_firmSettings.drift_alert_pct||5;
  } else {
    if(st){ st.textContent='Error saving'; st.style.color='var(--dn)'; }
  }
};

window.FIRM_exportAll = async function(){
  var r1 = await api('/api/enterprise/portfolios');
  var r2 = await api('/api/enterprise/clients');
  var r3 = await api('/api/enterprise/firm_settings');
  var dump = {
    exported_at: new Date().toISOString(),
    firm_settings: r3.ok?r3.d:{},
    portfolios: r1.ok?r1.d:[],
    clients: r2.ok?r2.d:[],
  };
  var blob = new Blob([JSON.stringify(dump,null,2)],{type:'application/json'});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'bloomberg_enterprise_export_'+new Date().toISOString().slice(0,10)+'.json';
  a.click();
};

// ── Morning Brief ─────────────────────────────────────────────────────────────
async function FIRM_loadBrief(){
  var r = await api('/api/enterprise/morning_brief');
  if(!r.ok) return;
  _briefData = r.d;
  FIRM_renderBrief();
}

function FIRM_renderBrief(){
  var d = _briefData;
  if(!d) return;
  var el = document.getElementById('firm-brief-content');
  if(!el) return;

  var firmName = _firmSettings.firm_name || 'BLOOMBERG / NER ENTERPRISE';
  var pnlC = d.grand_pnl>=0?'var(--up)':'var(--dn)';

  var html = '<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:16px">'+firmName+' · MORNING BRIEF · '+d.date+'</div>';

  // Headline KPIs
  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--bdr);border:1px solid var(--bdr);margin-bottom:16px">'
    +'<div style="background:var(--bg2);padding:10px 14px"><div style="font-size:8px;color:var(--txt3);margin-bottom:4px">TOTAL FIRM AUM</div><div style="font-size:20px;font-weight:700;color:var(--wht)">$'+fk(d.grand_aum)+'</div></div>'
    +'<div style="background:var(--bg2);padding:10px 14px"><div style="font-size:8px;color:var(--txt3);margin-bottom:4px">TOTAL P&L</div><div style="font-size:20px;font-weight:700;color:'+pnlC+'">'+(d.grand_pnl>=0?'+':'')+'$'+fk(Math.abs(d.grand_pnl))+'</div></div>'
    +'<div style="background:var(--bg2);padding:10px 14px"><div style="font-size:8px;color:var(--txt3);margin-bottom:4px">RETURN</div><div style="font-size:20px;font-weight:700;color:'+pnlC+'">'+(d.grand_pnl_pct>=0?'+':'')+d.grand_pnl_pct.toFixed(2)+'%</div></div>'
    +'<div style="background:var(--bg2);padding:10px 14px"><div style="font-size:8px;color:var(--txt3);margin-bottom:4px">PORTFOLIOS / CLIENTS</div><div style="font-size:20px;font-weight:700;color:var(--wht)">'+d.num_portfolios+' / '+d.num_clients+'</div></div>'
    +'</div>';

  // Action items summary
  var actions = d.overdue_reviews.length + d.flags.length + d.rebal_needed.length;
  html += '<div style="display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap">';
  html += _briefPill(d.overdue_reviews.length+' overdue reviews', d.overdue_reviews.length>0?'var(--yel)':'var(--up)');
  html += _briefPill(d.flags.length+' compliance flags', d.flags.length>0?'var(--dn)':'var(--up)');
  html += _briefPill(d.rebal_needed.length+' portfolios need rebalancing', d.rebal_needed.length>0?'var(--yel)':'var(--up)');
  html += '</div>';

  // Top movers
  if(d.top_movers.length){
    html += '<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:8px">TOP MOVERS TODAY</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:16px">';
    d.top_movers.slice(0,6).forEach(function(m){
      var c = m.pnl_pct>=0?'var(--up)':'var(--dn)';
      html += '<div class="brief-card '+(m.pnl_pct>=0?'ok':'danger')+'" style="display:flex;justify-content:space-between;align-items:center">'
        +'<span><b style="color:var(--cyn)">'+m.ticker+'</b> <span style="color:var(--txt3);font-size:9px">'+m.pf_name+'</span></span>'
        +'<span style="color:'+c+';font-weight:700">'+(m.pnl_pct>=0?'+':'')+m.pnl_pct.toFixed(2)+'%</span>'
        +'</div>';
    });
    html += '</div>';
  }

  // Overdue reviews
  if(d.overdue_reviews.length){
    html += '<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-bottom:8px">OVERDUE CLIENT REVIEWS</div>';
    d.overdue_reviews.forEach(function(r){
      html += '<div class="brief-card warn" style="display:flex;justify-content:space-between;align-items:center">'
        +'<span style="color:var(--wht);font-weight:700">'+r.client_name+'</span>'
        +'<span style="color:var(--yel)">'+r.days_overdue+'d overdue</span>'
        +'<span style="color:var(--txt3)">Last: '+r.last_review+'</span>'
        +'<button class="btn btn-xs" onclick="FIRM_quickReview(\''+r.client_id+'\',\''+r.client_name+'\')">LOG REVIEW</button>'
        +'</div>';
    });
  }

  // Compliance flags
  if(d.flags.length){
    html += '<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-top:12px;margin-bottom:8px">COMPLIANCE FLAGS</div>';
    d.flags.forEach(function(f){
      html += '<div class="brief-card danger"><b style="color:var(--dn)">'+f.type+'</b> '+f.pf+' — '+f.detail+'</div>';
    });
  }

  // Rebalancing
  if(d.rebal_needed.length){
    html += '<div style="font-size:9px;color:var(--txt3);letter-spacing:2px;margin-top:12px;margin-bottom:8px">REBALANCING NEEDED</div>';
    d.rebal_needed.forEach(function(r){
      var c = r.drift>0?'var(--up)':'var(--dn)';
      html += '<div class="brief-card warn" style="display:flex;justify-content:space-between">'
        +'<span><b style="color:var(--cyn)">'+r.ticker+'</b> in '+r.pf+'</span>'
        +'<span>Actual <b>'+r.actual+'%</b> vs Target <b>'+r.target+'%</b></span>'
        +'<span style="color:'+c+';font-weight:700">Drift '+(r.drift>0?'+':'')+r.drift+'%</span>'
        +'</div>';
    });
  }

  if(actions===0) html += '<div style="color:var(--up);font-size:11px;padding:20px 0">✓ All clear — no action items today.</div>';

  el.innerHTML = html;
}

function _briefPill(text, color){
  return '<div style="padding:4px 12px;background:var(--bg3);border:1px solid '+color+';color:'+color+';font-size:9px;font-weight:700">'+text+'</div>';
}

window.FIRM_quickReview = async function(cid, name){
  var notes = prompt('Quick review note for '+name+' (optional):','');
  if(notes===null) return;
  var r = await apiPost('/api/enterprise/clients/'+cid+'/review',{notes:notes});
  if(r.ok){
    var bar=document.getElementById('cmd-st');
    if(bar){bar.textContent='Review logged: '+name;bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);}
    await FIRM_loadBrief();
  }
};

// ── Main overview / exposure / model / compliance / revenue ───────────────────
async function FIRM_loadMain(){
  var r = await api('/api/enterprise/firm_analytics');
  if(!r.ok){ document.getElementById('firm-inner').innerHTML='<div style="padding:20px;color:var(--dn);font-size:11px">Failed to load firm analytics.</div>'; return; }
  _firmData = r.d;
  FIRM_renderOverview();
  FIRM_renderExposure();
  FIRM_renderModel();
  FIRM_renderCompliance();
  FIRM_renderRevenue();
  FIRM_renderRebalance();
}

function FIRM_renderOverview(){
  var d = _firmData; var s = d.summary;
  var kpis = document.getElementById('firm-kpis');
  if(kpis) kpis.innerHTML=[
    {l:'TOTAL AUM',  v:_fmtAum(s.grand_aum), c:'var(--wht)'},
    {l:'TOTAL P&L',  v:(s.grand_pnl>=0?'+':'')+'$'+fk(Math.abs(s.grand_pnl)), c:s.grand_pnl>=0?'var(--up)':'var(--dn)'},
    {l:'RETURN',     v:(s.grand_pnl_pct>=0?'+':'')+s.grand_pnl_pct.toFixed(2)+'%', c:s.grand_pnl_pct>=0?'var(--up)':'var(--dn)'},
    {l:'PORTFOLIOS', v:s.num_portfolios, c:'var(--wht)'},
    {l:'CLIENTS',    v:s.num_clients, c:'var(--cyn)'},
  ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');

  // AUM over time
  var ctx = document.getElementById('firm-aum-chart');
  if(ctx && d.aum_timeline.length>1){
    if(_firmCharts.aum) _firmCharts.aum.destroy();
    _firmCharts.aum = new Chart(ctx,{type:'line',data:{
      labels:d.aum_timeline.map(function(e){return e.date;}),
      datasets:[{data:d.aum_timeline.map(function(e){return e.aum;}),borderColor:'#ff8c00',borderWidth:2,pointRadius:0,fill:true,backgroundColor:'rgba(255,140,0,.07)'}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return ' AUM: $'+fk(c.parsed.y);}}}},
        scales:{x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#555',font:{size:9},maxTicksLimit:8}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#555',font:{size:9},callback:function(v){return '$'+fk(v);}}}}}});
  } else if(ctx){
    ctx.parentElement.innerHTML='<div style="padding:20px;color:var(--txt3);font-size:10px">Add cash deposits to portfolios to build AUM history.</div>';
  }

  // Pie
  var pctx = document.getElementById('firm-pf-pie');
  if(pctx && d.portfolios.length){
    if(_firmCharts.pie) _firmCharts.pie.destroy();
    var colors=['#ff8c00','#00e5ff','#00c853','#e040fb','#ffd600','#f44336','#534AB7','#00bfa5'];
    _firmCharts.pie = new Chart(pctx,{type:'doughnut',data:{
      labels:d.portfolios.map(function(p){return p.name;}),
      datasets:[{data:d.portfolios.map(function(p){return p.aum;}),backgroundColor:colors.slice(0,d.portfolios.length),borderColor:'#111',borderWidth:2}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{position:'right',labels:{color:'#888',font:{size:9},boxWidth:8}},tooltip:{...TT,callbacks:{label:function(c){return ' '+c.label+': $'+fk(c.parsed);}}}}}});
  }

  // Table
  var tbl = document.getElementById('firm-pf-table');
  if(tbl) tbl.innerHTML='<thead><tr><th>CLIENT</th><th>PORTFOLIO</th><th class="r">AUM</th><th class="r">VALUE</th><th class="r">CASH</th><th class="r">P&L</th><th class="r">RETURN</th><th>LAST ACTIVITY</th></tr></thead>'
    +'<tbody>'+d.portfolios.map(function(p){
      var pc=p.pnl>=0?'var(--up)':'var(--dn)';
      return '<tr><td style="color:var(--org)">'+p.client+'</td><td style="color:var(--wht);font-weight:600">'+p.name+'</td><td class="r" style="font-weight:700">$'+fk(p.aum)+'</td><td class="r">$'+fk(p.value)+'</td><td class="r" style="color:var(--cyn)">$'+fk(p.cash)+'</td><td class="r" style="color:'+pc+'">'+(p.pnl>=0?'+':'')+'$'+fk(Math.abs(p.pnl))+'</td><td class="r" style="color:'+pc+'">'+(p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(2)+'%</td><td style="color:var(--txt2)">'+p.last_activity+'</td></tr>';
    }).join('')+'</tbody>';
}

function FIRM_renderExposure(){
  var d = _firmData;
  var ctx = document.getElementById('firm-exposure-chart');
  if(ctx && d.top_exposures.length){
    if(_firmCharts.exp) _firmCharts.exp.destroy();
    _firmCharts.exp = new Chart(ctx,{type:'bar',data:{
      labels:d.top_exposures.map(function(e){return e.ticker;}),
      datasets:[{data:d.top_exposures.map(function(e){return e.pct_of_aum;}),backgroundColor:d.top_exposures.map(function(e){return e.pct_of_aum>15?'rgba(244,67,54,.7)':'rgba(255,140,0,.6)';}),borderWidth:0}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return ' '+c.parsed.y.toFixed(2)+'% of firm AUM';}}}},
        scales:{x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#888'}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#555',callback:function(v){return v.toFixed(0)+'%';}}}}}});
  }

  var flagEl = document.getElementById('firm-conc-flags');
  if(flagEl) flagEl.innerHTML = !d.concentration_flags.length
    ? '<div style="color:var(--up);font-size:10px">✓ No ticker exceeds 15% of firm AUM.</div>'
    : d.concentration_flags.map(function(f){return '<div style="border-left:3px solid var(--dn);padding:6px 10px;background:rgba(0,0,0,.2);margin-bottom:6px;font-size:10px"><b style="color:var(--dn)">'+f.ticker+'</b> — '+f.pct_of_aum.toFixed(1)+'% of firm AUM ($'+fk(f.value)+')</div>';}).join('');

  var tblEl = document.getElementById('firm-exposure-table');
  if(tblEl) tblEl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>TICKER</th><th class="r">TOTAL VALUE</th><th class="r">% OF FIRM AUM</th></tr></thead><tbody>'
    +d.top_exposures.map(function(e){var c=e.pct_of_aum>15?'var(--dn)':e.pct_of_aum>8?'var(--yel)':'var(--txt2)';return '<tr><td style="color:var(--cyn);font-weight:700">'+e.ticker+'</td><td class="r">$'+fk(e.value)+'</td><td class="r" style="color:'+c+'">'+e.pct_of_aum.toFixed(2)+'%</td></tr>';}).join('')+'</tbody></table>';
}

function FIRM_renderRebalance(){
  if(!_briefData&&!_firmData) return;
  var rebal = _briefData?_briefData.rebal_needed:[];
  var el = document.getElementById('firm-rebal-list');
  if(!el) return;
  if(!rebal.length){
    el.innerHTML='<div style="color:var(--up);font-size:10px">✓ All portfolios within drift threshold.</div>';
    return;
  }
  el.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>PORTFOLIO</th><th>TICKER</th><th class="r">ACTUAL</th><th class="r">TARGET</th><th class="r">DRIFT</th><th class="r">ACTION</th></tr></thead><tbody>'
    +rebal.map(function(r){
      var dc=r.drift>0?'var(--up)':'var(--dn)';
      var action=r.drift>0?'TRIM':'ADD';
      return '<tr><td style="color:var(--wht)">'+r.pf+'</td><td style="color:var(--cyn);font-weight:700">'+r.ticker+'</td><td class="r">'+r.actual+'%</td><td class="r">'+r.target+'%</td><td class="r" style="color:'+dc+';font-weight:700">'+(r.drift>0?'+':'')+r.drift+'%</td><td class="r" style="color:'+dc+'">'+action+'</td></tr>';
    }).join('')+'</tbody></table>';
}

// ── Client Reviews ────────────────────────────────────────────────────────────
async function FIRM_renderReviews(){
  var r = await api('/api/enterprise/clients');
  if(!r.ok) return;
  var clients = r.d;
  var overdue = _briefData?_briefData.overdue_reviews:[];

  var overdueEl = document.getElementById('firm-overdue-reviews');
  if(overdueEl){
    if(!overdue.length){
      overdueEl.innerHTML='<div style="color:var(--up);font-size:10px">✓ No overdue reviews.</div>';
    } else {
      overdueEl.innerHTML=overdue.map(function(rv){
        return '<div class="review-row">'
          +'<span style="color:var(--wht);font-weight:700;width:160px">'+rv.client_name+'</span>'
          +'<span style="color:var(--yel)">'+rv.days_overdue+'d overdue</span>'
          +'<span style="color:var(--txt3)">Last: '+rv.last_review+'</span>'
          +'<span style="color:var(--txt3)">('+rv.frequency+')</span>'
          +'<button class="btn on btn-xs" style="margin-left:auto" onclick="FIRM_quickReview(\''+rv.client_id+'\',\''+rv.client_name+'\')">LOG REVIEW</button>'
          +'</div>';
      }).join('');
    }
  }

  var allEl = document.getElementById('firm-all-reviews');
  if(allEl){
    allEl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>CLIENT</th><th>RISK</th><th>FREQUENCY</th><th>LAST REVIEW</th><th class="r">STATUS</th><th class="r">ACTION</th></tr></thead><tbody>'
      +clients.map(function(c){
        var last = c.last_review_date||'Never';
        var freq = c.review_frequency||(_firmSettings.default_review_freq||'quarterly');
        var freqDays = {monthly:30,quarterly:90,annual:365}[freq]||90;
        var status='UP TO DATE'; var statusC='var(--up)';
        if(last==='Never'){ status='NEVER REVIEWED'; statusC='var(--dn)'; }
        else {
          try{
            var days=Math.floor((Date.now()-new Date(last+'T00:00:00Z').getTime())/86400000);
            if(days>=freqDays){ status='OVERDUE '+Math.max(0,days-freqDays)+'d'; statusC='var(--yel)'; }
          }catch(e){}
        }
        var riskC={conservative:'var(--up)',moderate:'var(--yel)',aggressive:'var(--dn)'}[c.risk_profile]||'var(--txt2)';
        return '<tr><td style="color:var(--wht);font-weight:600">'+c.name+'</td>'
          +'<td style="color:'+riskC+'">'+((c.risk_profile||'—').toUpperCase())+'</td>'
          +'<td style="color:var(--txt2)">'+freq+'</td>'
          +'<td style="color:var(--txt2)">'+last+'</td>'
          +'<td class="r" style="color:'+statusC+';font-weight:700">'+status+'</td>'
          +'<td class="r"><button class="btn btn-xs" onclick="FIRM_quickReview(\''+c.id+'\',\''+c.name+'\')">LOG</button></td>'
          +'</tr>';
      }).join('')+'</tbody></table>';
  }
}

// ── Realized P&L ──────────────────────────────────────────────────────────────
async function FIRM_loadRealized(){
  var r = await api('/api/enterprise/firm_realized');
  if(!r.ok) return;
  _realizedData = r.d;

  var kpis = document.getElementById('firm-real-kpis');
  if(kpis) kpis.innerHTML=[
    {l:'TOTAL REALIZED P&L', v:(r.d.total_realized>=0?'+':'')+'$'+fk(Math.abs(r.d.total_realized)), c:r.d.total_realized>=0?'var(--up)':'var(--dn)'},
    {l:'DIVIDEND INCOME',    v:'+$'+fk(r.d.total_dividends), c:'var(--up)'},
    {l:'TOTAL INCOME',       v:'$'+fk(r.d.total_income), c:'var(--up)'},
  ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');

  // Cumulative chart
  var ctx = document.getElementById('firm-real-chart');
  if(ctx && r.d.trades.length){
    if(_firmCharts.real) _firmCharts.real.destroy();
    var sorted = r.d.trades.filter(function(t){return t.close_date;}).sort(function(a,b){return a.close_date.localeCompare(b.close_date);});
    var running=0; var pts=[]; var labels=[];
    sorted.forEach(function(t){running+=t.realised_pnl;pts.push(round2(running));labels.push(t.close_date);});
    function round2(v){return Math.round(v*100)/100;}
    _firmCharts.real = new Chart(ctx,{type:'line',data:{labels:labels,datasets:[{data:pts,borderColor:running>=0?'#00c853':'#f44336',borderWidth:1.8,pointRadius:0,fill:true,backgroundColor:running>=0?'rgba(0,200,83,.07)':'rgba(244,67,54,.06)'}]},
      options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false},tooltip:{...TT,callbacks:{label:function(c){return ' Cumulative: $'+fk(c.parsed.y);}}}},
        scales:{x:{grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#555',font:{size:9},maxTicksLimit:8}},y:{grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#555',callback:function(v){return '$'+fk(v);}}}}}});
  }

  var tbl = document.getElementById('firm-real-table');
  if(tbl) tbl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>DATE</th><th>PORTFOLIO</th><th>CLIENT</th><th>TICKER</th><th class="r">P&L</th><th>NOTES</th></tr></thead><tbody>'
    +r.d.trades.map(function(t){var pc=t.realised_pnl>=0?'var(--up)':'var(--dn)';var isDiv=t.type==='dividend';return '<tr><td style="color:var(--txt2)">'+(t.close_date||'—')+'</td><td>'+t.pf_name+'</td><td style="color:var(--org)">'+t.client+'</td><td style="color:var(--cyn);font-weight:700">'+t.ticker+(isDiv?' <span style="font-size:8px;color:var(--yel)">DIV</span>':'')+'</td><td class="r" style="color:'+pc+';font-weight:700">'+(t.realised_pnl>=0?'+':'')+'$'+f(t.realised_pnl,2)+'</td><td style="color:var(--txt3)">'+(t.notes||'')+'</td></tr>';}).join('')+'</tbody></table>';
}

// ── Attribution ───────────────────────────────────────────────────────────────
window.FIRM_loadAttribution = async function(){
  var days = (document.getElementById('attr-days')||{}).value||'180';
  var r = await api('/api/enterprise/firm_attribution?days='+days);
  if(!r.ok) return;
  _attrData = r.d;

  if(!r.d.months.length){
    var el=document.getElementById('firm-attr-table');
    if(el) el.innerHTML='<div style="color:var(--txt3);font-size:10px;padding:10px">Need at least 30 days of price history and closed positions to compute attribution.</div>';
    return;
  }

  // Stacked bar chart
  var ctx = document.getElementById('firm-attr-chart');
  if(ctx && r.d.months.length){
    if(_firmCharts.attr) _firmCharts.attr.destroy();
    var colors=['#ff8c00','#00e5ff','#00c853','#e040fb','#ffd600','#f44336','#534AB7'];
    var datasets = r.d.pf_names.map(function(name,i){
      return {label:name,data:r.d.months.map(function(m){return r.d.data[m][name]||0;}),backgroundColor:colors[i%colors.length],borderWidth:0};
    });
    _firmCharts.attr = new Chart(ctx,{type:'bar',data:{labels:r.d.months,datasets:datasets},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{labels:{color:'#888',font:{size:9}}},tooltip:{...TT}},
        scales:{x:{stacked:true,grid:{color:'rgba(46,46,46,.3)'},ticks:{color:'#555',font:{size:9}}},y:{stacked:true,grid:{color:'#1a1a1a'},position:'right',ticks:{color:'#555',font:{size:9}}}}}});
  }

  // Table
  var tbl = document.getElementById('firm-attr-table');
  if(tbl){
    var ths = '<tr><th>MONTH</th>'+r.d.pf_names.map(function(n){return '<th class="r">'+n+'</th>';}).join('')+'<th class="r">TOTAL</th></tr>';
    var trs = r.d.months.map(function(m){
      var total=0;
      var cells=r.d.pf_names.map(function(n){var v=r.d.data[m][n]||0;total+=v;var c=v>=0?'var(--up)':'var(--dn)';return '<td class="r" style="color:'+c+'">'+(v>=0?'+':'')+v.toFixed(3)+'%</td>';}).join('');
      var tc=total>=0?'var(--up)':'var(--dn)';
      return '<tr><td style="color:var(--txt2)">'+m+'</td>'+cells+'<td class="r" style="color:'+tc+';font-weight:700">'+(total>=0?'+':'')+total.toFixed(3)+'%</td></tr>';
    }).join('');
    tbl.innerHTML='<table class="dt" style="width:100%;min-width:500px"><thead>'+ths+'</thead><tbody>'+trs+'</tbody></table>';
  }
};

// ── Model Portfolio ───────────────────────────────────────────────────────────
function FIRM_renderModel(){
  var d = _firmData;
  var editor = document.getElementById('firm-model-editor');
  if(!editor) return;
  var savedModel={};
  try{savedModel=JSON.parse(localStorage.getItem('firm_model')||'{}');}catch(e){}
  var tickers=d.top_exposures.map(function(e){return e.ticker;});
  editor.innerHTML='<div style="font-size:9px;color:var(--txt3);margin-bottom:10px">Set target weights. Portfolios will be compared against this model.</div>'
    +'<table class="dt" style="width:100%;max-width:500px"><thead><tr><th>TICKER</th><th class="r">TARGET WEIGHT %</th></tr></thead><tbody>'
    +tickers.map(function(t){return '<tr><td style="color:var(--cyn)">'+t+'</td><td class="r"><input type="number" value="'+(savedModel[t]||0)+'" step="1" min="0" max="100" style="width:70px;background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:2px 4px;outline:none" data-ticker="'+t+'"></td></tr>';}).join('')+'</tbody></table>';
  document.getElementById('firm-model-drift').innerHTML='<div style="color:var(--txt3);font-size:10px">Save model to see drift.</div>';
}

window.FIRM_saveModel = function(){
  var model={};
  document.querySelectorAll('#firm-model-editor input[type=number]').forEach(function(inp){if(inp.dataset.ticker)model[inp.dataset.ticker]=parseFloat(inp.value)||0;});
  try{localStorage.setItem('firm_model',JSON.stringify(model));}catch(e){}
  document.getElementById('firm-model-drift').innerHTML='<div style="color:var(--up);font-size:10px">✓ Model saved. Drift visible in REBALANCING QUEUE tab.</div>';
  var bar=document.getElementById('cmd-st');
  if(bar){bar.textContent='Model portfolio saved';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);}
};

// ── Compliance / Revenue ──────────────────────────────────────────────────────
function FIRM_renderCompliance(){
  var el=document.getElementById('firm-compliance-list');
  if(!el||!_firmData) return;
  var flags=_firmData.compliance;
  if(!flags.length){el.innerHTML='<div style="color:var(--up);font-size:10px;padding:8px">✓ No compliance issues detected.</div>';return;}
  var typeC={INACTIVE:'var(--yel)',CASH_ONLY:'var(--yel)',CONCENTRATION:'var(--dn)'};
  el.innerHTML=flags.map(function(f){return '<div class="compliance-flag'+(f.type==='CONCENTRATION'?' critical':'')+'"><span style="color:'+(typeC[f.type]||'var(--yel)')+';font-weight:700;font-size:9px;letter-spacing:1px">'+f.type+'</span> <span style="color:var(--wht)">'+f.pf+'</span> — <span style="color:var(--txt2)">'+f.detail+'</span></div>';}).join('');
}

function FIRM_renderRevenue(){
  var d=_firmData; var s=d.summary;
  var fee = parseFloat(_firmSettings.mgmt_fee)||1;  // %/month
  var sym = _ccy();
  var kpis=document.getElementById('firm-rev-kpis');
  if(kpis) kpis.innerHTML=[
    {l:'TOTAL AUM',v:_fmtAum(s.grand_aum),c:'var(--wht)'},
    {l:'EST. MRR ('+fee+'%/mo)',v:_fmtAum(s.grand_aum*fee/100),c:'var(--up)'},
    {l:'EST. ARR (x12)',v:_fmtAum(s.grand_aum*fee/100*12),c:'var(--up)'},
    {l:'AVG AUM / PORTFOLIO',v:s.num_portfolios?_fmtAum(s.grand_aum/s.num_portfolios):'—',c:'var(--cyn)'},
  ].map(function(k){return '<div class="sc"><div class="sc-l">'+k.l+'</div><div class="sc-v" style="color:'+k.c+'">'+k.v+'</div></div>';}).join('');

  var tbl=document.getElementById('firm-rev-table');
  if(tbl) tbl.innerHTML='<table class="dt" style="width:100%"><thead><tr><th>PORTFOLIO</th><th>CLIENT</th><th class="r">AUM</th><th class="r">MRR</th><th class="r">ARR</th></tr></thead><tbody>'
    +d.portfolios.map(function(p){
      var mrr=p.aum*fee/100; var arr=mrr*12;
      return '<tr><td style="color:var(--wht);font-weight:600">'+p.name+'</td><td style="color:var(--org)">'+p.client+'</td><td class="r">'+_fmtAum(p.aum)+'</td><td class="r" style="color:var(--up)">'+_fmtAum(mrr)+'</td><td class="r" style="color:var(--up)">'+_fmtAum(arr)+'</td></tr>';
    }).join('')+'</tbody></table>';
}

})();
