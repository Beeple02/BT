// BT Real Estate Space — realestate.js
// 4 IIFEs: PRICER · TRACKER · COMPARE · PORTFOLIO

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers (not in IIFE — tiny, safe to share)
// ─────────────────────────────────────────────────────────────────────────────
function re_fmt(n){
  if(n==null||n===undefined) return '—';
  if(typeof n!=='number') n=parseFloat(n);
  if(isNaN(n)) return '—';
  if(n>=1e6) return (n/1e6).toFixed(2)+'M';
  if(n>=1e3) return (n/1e3).toFixed(1)+'K';
  return n.toLocaleString();
}
function re_fmtFull(n){
  if(n==null||n===undefined) return '—';
  if(typeof n!=='number') n=parseFloat(n);
  if(isNaN(n)) return '—';
  return n.toLocaleString(undefined,{maximumFractionDigits:0});
}
function re_confClass(c){
  if(c==null) return '';
  if(c>=75) return 'conf-hi';
  if(c>=50) return 'conf-mid';
  return 'conf-lo';
}
function re_pnlClass(v){ return v>0?'up':v<0?'dn':''; }
function re_ago(ts){
  if(!ts) return '—';
  const s=Math.floor((Date.now()-ts)/1000);
  if(s<60) return s+'s ago';
  if(s<3600) return Math.floor(s/60)+'m ago';
  return Math.floor(s/3600)+'h ago';
}

// ─────────────────────────────────────────────────────────────────────────────
// PRICER IIFE
// ─────────────────────────────────────────────────────────────────────────────
(function(){
  var _mode = 'exact'; // 'exact' | 'smart'
  var _last = null;

  window.RE_PRICER_setMode = function(m){
    _mode = m;
    document.getElementById('rp-mode-exact').classList.toggle('on', m==='exact');
    document.getElementById('rp-mode-smart').classList.toggle('on', m==='smart');
    var inp = document.getElementById('rp-input');
    if(inp){
      inp.placeholder = m==='exact'
        ? 'Plot ID (e.g. RH011)…'
        : 'Describe a parcel (e.g. "RH corner plot near spawn")…';
    }
  };

  window.RE_PRICER_init = function(){
    // Ensure mode buttons reflect state
    RE_PRICER_setMode(_mode);
    // If we have a cached last result from session, re-render
    if(_last) _render(_last);
  };

  window.RE_PRICER_price = async function(){
    var raw = (document.getElementById('rp-input')||{}).value||'';
    raw = raw.trim();
    if(!raw) return;
    _setLoading(true);
    var body = {mode:'external-estimate'};
    if(_mode==='smart'){
      body.message = raw;
    } else {
      body.propertyId = raw;
    }
    var r = await apiPost('/api/realestate/price', body);
    _setLoading(false);
    if(!r.ok){
      _showError(r.d && (r.d.error||r.d.detail) || 'Pricing failed (HTTP '+r.s+')');
      // Show suggestions if present
      if(r.d && r.d.suggestions && r.d.suggestions.length){
        var el=document.getElementById('rp-error');
        if(el) el.innerHTML+='<br><span style="color:var(--txt3)">Suggestions: </span>'+r.d.suggestions.join(', ');
      }
      return;
    }
    var data = r.d.data || r.d;
    _last = data;
    _render(data);
  };

  // Load a specific plot ID from outside (e.g. tracker click)
  window.RE_PRICER_loadPlot = async function(plotId){
    var inp = document.getElementById('rp-input');
    if(inp) inp.value = plotId;
    RE_PRICER_setMode('exact');
    await RE_PRICER_price();
    // Switch to pricer space
    if(typeof switchSpace==='function') switchSpace('pricer');
  };

  function _setLoading(on){
    var ph=document.getElementById('rp-placeholder');
    var res=document.getElementById('rp-result');
    var ld=document.getElementById('rp-loading');
    var err=document.getElementById('rp-error');
    if(ph)  ph.style.display  = on?'none':'';
    if(res) res.style.display = on?'none':'';
    if(ld)  ld.style.display  = on?'block':'none';
    if(err) err.style.display = 'none';
  }

  function _showError(msg){
    var ph=document.getElementById('rp-placeholder');
    var res=document.getElementById('rp-result');
    var ld=document.getElementById('rp-loading');
    var err=document.getElementById('rp-error');
    if(ph)  ph.style.display='none';
    if(res) res.style.display='none';
    if(ld)  ld.style.display='none';
    if(err){ err.style.display='block'; err.textContent=msg; }
  }

  function _set(id,val){ var el=document.getElementById(id); if(el) el.textContent=val||'—'; }
  function _setHtml(id,val){ var el=document.getElementById(id); if(el) el.innerHTML=val||''; }

  function _render(d){
    // Show result, hide placeholder/loading/error
    var ph=document.getElementById('rp-placeholder');
    var res=document.getElementById('rp-result');
    var ld=document.getElementById('rp-loading');
    var err=document.getElementById('rp-error');
    if(ph)  ph.style.display='none';
    if(ld)  ld.style.display='none';
    if(err) err.style.display='none';
    if(res){ res.style.display='flex'; }

    var est = d.estimate||{};
    var hist= d.history||{};
    var res2= d.resolution;

    // Hero
    _set('rp-hero-id',  d.label||d.propertyId);
    _set('rp-hero-type', d.propertyType||'—');
    _set('rp-hero-area', d.areaGroup||'—');
    var mapEl=document.getElementById('rp-map-link');
    if(mapEl && d.mapLink){ mapEl.href=d.mapLink; mapEl.style.display=''; }
    else if(mapEl){ mapEl.style.display='none'; }

    // KPI strip
    _set('rp-area',    d.landArea!=null ? d.landArea.toLocaleString()+' m²':'—');
    _set('rp-dist',    d.distanceFromSpawn!=null ? d.distanceFromSpawn.toLocaleString()+' blocks':'—');
    _set('rp-coords',  d.x!=null&&d.z!=null ? 'X'+d.x+' Z'+d.z:'—');
    _set('rp-plotcount', d.plotCount!=null ? d.plotCount+' plot'+(d.plotCount===1?'':'s'):'—');

    // Estimate cards
    _set('rp-est-val',  est.estimate!=null?re_fmtFull(est.estimate):'—');
    _set('rp-est-lo',   est.estimateBandLow!=null?re_fmtFull(est.estimateBandLow):'—');
    _set('rp-est-hi',   est.estimateBandHigh!=null?re_fmtFull(est.estimateBandHigh):'—');
    var confEl=document.getElementById('rp-est-conf');
    if(confEl){
      confEl.textContent = est.confidence!=null ? est.confidence+'%':'—';
      confEl.className='rp-ec-v '+re_confClass(est.confidence);
    }

    // Resolution box
    var resBox=document.getElementById('rp-res-box');
    var badge=document.getElementById('rp-method-badge');
    var resText=document.getElementById('rp-res-text');
    if(res2 && resBox){
      resBox.style.display='flex';
      if(badge){
        badge.textContent=res2.method==='gemini'?'GEMINI MATCH':'EXACT MATCH';
        badge.className='rp-method-badge '+(res2.method==='gemini'?'gemini':'exact');
      }
      if(resText){
        var conf2=res2.confidence!=null?' ('+res2.confidence+'% confidence)':'';
        resText.textContent=(res2.reasoning||'')+conf2;
      }
    } else if(resBox){
      resBox.style.display='none';
    }

    // History
    _set('rp-hist-total', hist.count!=null?hist.count:'—');
    _set('rp-hist-list',  hist.listingCount!=null?hist.listingCount:'—');
    _set('rp-hist-conf',  hist.confirmedCount!=null?hist.confirmedCount:'—');
    _set('rp-hist-priv',  hist.confidentialCount!=null?hist.confidentialCount:'—');

    // Reasons
    var rDiv=document.getElementById('rp-reasons');
    var rSec=document.getElementById('rp-reasons-section');
    var reasons=(est.reasons||[]).concat(est.modelNotes||[]);
    if(rDiv){
      if(reasons.length){
        rDiv.innerHTML=reasons.map(function(r){return '<div class="rp-reason">'+_esc(r)+'</div>';}).join('');
        if(rSec) rSec.style.display='';
      } else {
        if(rSec) rSec.style.display='none';
      }
    }

    // Comps
    var compsBody=document.getElementById('rp-comps-body');
    var compsSec=document.getElementById('rp-comps-section');
    var comps=est.compsUsed||[];
    if(compsBody){
      if(comps.length){
        var html='<table class="dt" style="width:100%"><thead><tr>';
        // Build headers from first comp keys
        var keys=Object.keys(comps[0]).filter(function(k){return k!=='id';});
        keys.forEach(function(k){ html+='<th>'+_esc(k.toUpperCase())+'</th>'; });
        html+='</tr></thead><tbody>';
        comps.forEach(function(c){
          html+='<tr>';
          keys.forEach(function(k){ html+='<td>'+_esc(String(c[k]!=null?c[k]:'—'))+'</td>'; });
          html+='</tr>';
        });
        html+='</tbody></table>';
        compsBody.innerHTML=html;
        if(compsSec) compsSec.style.display='';
      } else {
        compsBody.innerHTML='<div class="rp-no-comps">No public comparable sales available for this parcel.</div>';
        if(compsSec) compsSec.style.display='';
      }
    }
  }

  function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
})();


// ─────────────────────────────────────────────────────────────────────────────
// TRACKER IIFE
// ─────────────────────────────────────────────────────────────────────────────
(function(){
  var LS_KEY = 're_tracker';
  var _list  = [];  // [{plotId, data, fetchedAt, error}]
  var _refreshing = false;

  function _load(){ try{ _list=JSON.parse(localStorage.getItem(LS_KEY)||'[]'); }catch(e){ _list=[]; } }
  function _save(){ try{ localStorage.setItem(LS_KEY,JSON.stringify(_list)); }catch(e){} }

  window.RE_TRACKER_init = function(){
    _load();
    _render();
  };

  window.RE_TRACKER_add = async function(){
    var inp=document.getElementById('rt-input');
    var raw=(inp&&inp.value||'').trim().toUpperCase();
    if(!raw) return;
    if(_list.find(function(x){return x.plotId===raw;})){
      _setStatus('Already tracked: '+raw); return;
    }
    inp.value='';
    _list.push({plotId:raw, data:null, fetchedAt:null, error:null});
    _save();
    _render();
    // Auto-price it
    await _fetchOne(_list.length-1);
    _render();
    _updateAgg();
  };

  window.RE_TRACKER_refreshAll = async function(){
    if(_refreshing) return;
    _refreshing=true;
    var btn=document.getElementById('rt-refresh-btn');
    if(btn) btn.disabled=true;
    _setStatus('Refreshing '+_list.length+' plots…');
    // Sequential to respect rate limit
    for(var i=0;i<_list.length;i++){
      await _fetchOne(i);
      _render();
      _updateAgg();
      _setStatus('Refreshed '+(i+1)+'/'+_list.length+'…');
      if(i<_list.length-1) await _sleep(700); // ~90 req/min limit buffer
    }
    _setStatus('');
    if(btn) btn.disabled=false;
    _refreshing=false;
  };

  window.RE_TRACKER_remove = function(plotId){
    _list=_list.filter(function(x){return x.plotId!==plotId;});
    _save();
    _render();
    _updateAgg();
  };

  async function _fetchOne(idx){
    var item=_list[idx];
    if(!item) return;
    item.error=null;
    var r=await apiPost('/api/realestate/price',{mode:'external-estimate',propertyId:item.plotId});
    if(r.ok){
      item.data=(r.d.data||r.d);
      item.fetchedAt=Date.now();
      item.error=null;
    } else {
      item.error=(r.d&&(r.d.error||r.d.detail))||'Error '+r.s;
      item.fetchedAt=Date.now();
    }
    _save();
  }

  function _render(){
    var empty=document.getElementById('rt-empty');
    var tbl=document.getElementById('rt-table');
    var tbody=document.getElementById('rt-tbody');
    if(!tbody) return;
    if(!_list.length){
      if(empty) empty.style.display='';
      if(tbl)   tbl.style.display='none';
      _updateAgg();
      return;
    }
    if(empty) empty.style.display='none';
    if(tbl)   tbl.style.display='';
    var html='';
    _list.forEach(function(item){
      var d=item.data;
      var est=d&&d.estimate||{};
      var hist=d&&d.history||{};
      var conf=est.confidence;
      var confCls=re_confClass(conf);
      var age=re_ago(item.fetchedAt);
      var stale=item.fetchedAt&&(Date.now()-item.fetchedAt>600000);
      html+='<tr onclick="if(window.RE_PRICER_loadPlot)RE_PRICER_loadPlot(\''+_esc(item.plotId)+'\')" style="cursor:pointer">';
      html+='<td><span class="rt-dot '+(item.error?'err':d?'ok':'loading')+'"></span></td>';
      html+='<td style="font-weight:700;color:var(--wht)">'+_esc(item.plotId)+'</td>';
      html+='<td style="color:var(--txt2)">'+(d&&d.propertyType||'—')+'</td>';
      html+='<td style="color:var(--txt2)">'+(d&&d.areaGroup||item.error||'—')+'</td>';
      html+='<td class="r">'+(d&&d.landArea!=null?d.landArea.toLocaleString():'—')+'</td>';
      html+='<td class="r" style="color:var(--org)">'+(est.estimate!=null?re_fmtFull(est.estimate):'—')+'</td>';
      html+='<td class="r">'+(est.estimateBandLow!=null?re_fmtFull(est.estimateBandLow):'—')+'</td>';
      html+='<td class="r">'+(est.estimateBandHigh!=null?re_fmtFull(est.estimateBandHigh):'—')+'</td>';
      html+='<td class="r"><span class="'+confCls+'">'+(conf!=null?conf+'%':'—')+'</span></td>';
      html+='<td class="r">'+(hist.found?hist.count:'—')+'</td>';
      html+='<td><span class="rt-age'+(stale?' stale':'')+'">'+age+'</span></td>';
      html+='<td onclick="event.stopPropagation();RE_TRACKER_remove(\''+_esc(item.plotId)+'\')"><span class="rt-rm">×</span></td>';
      html+='</tr>';
    });
    tbody.innerHTML=html;
    _updateAgg();
  }

  function _updateAgg(){
    var priced=_list.filter(function(x){return x.data&&x.data.estimate&&x.data.estimate.estimate!=null;});
    var total=priced.reduce(function(s,x){return s+(x.data.estimate.estimate||0);},0);
    var avg=priced.length?Math.round(total/priced.length):null;
    var avgConf=priced.length?Math.round(priced.reduce(function(s,x){return s+(x.data.estimate.confidence||0);},0)/priced.length):null;
    _set('rt-agg-count',_list.length);
    _set('rt-agg-total',priced.length?re_fmtFull(total):'—');
    _set('rt-agg-avg',  avg!=null?re_fmtFull(avg):'—');
    _set('rt-agg-conf', avgConf!=null?avgConf+'%':'—');
  }

  function _setStatus(msg){ var el=document.getElementById('rt-status'); if(el) el.textContent=msg; }
  function _set(id,v){ var el=document.getElementById(id); if(el) el.textContent=v||'—'; }
  function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&#39;'); }
  function _sleep(ms){ return new Promise(function(r){setTimeout(r,ms);}); }
})();


// ─────────────────────────────────────────────────────────────────────────────
// COMPARE IIFE
// ─────────────────────────────────────────────────────────────────────────────
(function(){
  var _chart = null;

  window.RE_COMPARE_init = function(){
    // nothing to restore
  };

  window.RE_COMPARE_run = async function(){
    var ids=[];
    for(var i=1;i<=6;i++){
      var el=document.getElementById('rc-p'+i);
      var v=el&&el.value.trim();
      if(v) ids.push(v.toUpperCase());
    }
    ids=[...new Set(ids)]; // dedup
    if(ids.length<2){ _setStatus('Enter at least 2 plot IDs'); return; }
    _setStatus('Pricing '+ids.length+' plots…');

    var ph=document.getElementById('rc-placeholder');
    var res=document.getElementById('rc-result');
    if(ph) ph.style.display='none';
    if(res) res.style.display='none';

    // Fetch all (sequential to be safe with rate limits)
    var results=[];
    for(var j=0;j<ids.length;j++){
      _setStatus('Pricing '+ids[j]+'… ('+(j+1)+'/'+ids.length+')');
      var r=await apiPost('/api/realestate/price',{mode:'external-estimate',propertyId:ids[j]});
      results.push({id:ids[j], ok:r.ok, d:r.ok?(r.d.data||r.d):null, err:r.ok?null:(r.d&&(r.d.error||r.d.detail)||'Error '+r.s)});
      if(j<ids.length-1) await _sleep(500);
    }
    _setStatus('');
    _renderGrid(results);
    _renderChart(results);
    if(res) res.style.display='flex';
  };

  function _renderGrid(results){
    var head=document.getElementById('rc-grid-head');
    var body=document.getElementById('rc-grid-body');
    if(!head||!body) return;

    // Header row: row label col + one col per plot
    var h='<th class="row-lbl">METRIC</th>';
    results.forEach(function(r){
      h+='<th>'+_esc(r.id)+'</th>';
    });
    head.innerHTML=h;

    // Extract estimates for best/worst highlighting
    var ests=results.map(function(r){return r.d&&r.d.estimate?r.d.estimate.estimate:null;});
    var validEsts=ests.filter(function(e){return e!=null;});
    var maxEst=validEsts.length?Math.max.apply(null,validEsts):null;
    var minEst=validEsts.length?Math.min.apply(null,validEsts):null;

    var ROWS=[
      {label:'ESTIMATE',      fn:function(d){return d&&d.estimate&&d.estimate.estimate!=null?re_fmtFull(d.estimate.estimate):'—';}, highlight:true},
      {label:'BAND LOW',      fn:function(d){return d&&d.estimate&&d.estimate.estimateBandLow!=null?re_fmtFull(d.estimate.estimateBandLow):'—';}},
      {label:'BAND HIGH',     fn:function(d){return d&&d.estimate&&d.estimate.estimateBandHigh!=null?re_fmtFull(d.estimate.estimateBandHigh):'—';}},
      {label:'CONFIDENCE',    fn:function(d){return d&&d.estimate&&d.estimate.confidence!=null?d.estimate.confidence+'%':'—';}},
      {label:'TYPE',          fn:function(d){return d&&d.propertyType||'—';}},
      {label:'AREA GROUP',    fn:function(d){return d&&d.areaGroup||'—';}},
      {label:'LAND AREA',     fn:function(d){return d&&d.landArea!=null?d.landArea.toLocaleString()+' m²':'—';}},
      {label:'DIST FROM SPAWN', fn:function(d){return d&&d.distanceFromSpawn!=null?d.distanceFromSpawn.toLocaleString()+' blocks':'—';}},
      {label:'HISTORY',       fn:function(d){return d&&d.history?d.history.count+' records':'—';}},
    ];

    var html='';
    ROWS.forEach(function(row){
      html+='<tr><td class="row-lbl">'+row.label+'</td>';
      results.forEach(function(r,i){
        var val=row.fn(r.d);
        var cls='';
        if(row.highlight && r.d && r.d.estimate && r.d.estimate.estimate!=null){
          var v=r.d.estimate.estimate;
          if(validEsts.length>1 && v===maxEst) cls='best';
          else if(validEsts.length>1 && v===minEst) cls='worst';
        }
        html+='<td class="'+cls+'">'+_esc(String(val))+'</td>';
      });
      html+='</tr>';
    });
    body.innerHTML=html;
  }

  function _renderChart(results){
    var canvas=document.getElementById('rc-chart');
    if(!canvas) return;
    _chart = dc(_chart);
    var labels=results.map(function(r){return r.id;});
    var data=results.map(function(r){return r.d&&r.d.estimate?r.d.estimate.estimate:0;});
    var colors=data.map(function(v,i){return i===0?'rgba(255,140,0,.7)':'rgba(0,229,255,.5)';});
    _chart=mkBar(canvas, labels, data, colors);
  }

  function _setStatus(msg){ var el=document.getElementById('rc-status'); if(el) el.textContent=msg; }
  function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
  function _sleep(ms){ return new Promise(function(r){setTimeout(r,ms);}); }
})();


// ─────────────────────────────────────────────────────────────────────────────
// PORTFOLIO IIFE
// ─────────────────────────────────────────────────────────────────────────────
(function(){
  var LS_KEY   = 're_portfolio';
  var _pos     = [];  // [{plotId, purchasePrice, purchaseDate, addedAt, data, fetchedAt, error}]
  var _chart   = null;
  var _refreshing = false;

  function _load(){ try{ _pos=JSON.parse(localStorage.getItem(LS_KEY)||'[]'); }catch(e){ _pos=[]; } }
  function _save(){ try{ localStorage.setItem(LS_KEY,JSON.stringify(_pos)); }catch(e){} }

  window.RE_PORTFOLIO_init = function(){
    _load();
    // Set today as default date
    var dateEl=document.getElementById('rpf-date');
    if(dateEl && !dateEl.value) dateEl.value=new Date().toISOString().slice(0,10);
    _render();
  };

  window.RE_PORTFOLIO_add = async function(){
    var id=(document.getElementById('rpf-id')||{}).value||'';
    id=id.trim().toUpperCase();
    var price=parseFloat((document.getElementById('rpf-price')||{}).value||'0')||0;
    var date=(document.getElementById('rpf-date')||{}).value||new Date().toISOString().slice(0,10);
    if(!id){ _setStatus('Enter a plot ID'); return; }
    if(_pos.find(function(x){return x.plotId===id;})){
      _setStatus('Already in portfolio: '+id); return;
    }
    var pos={plotId:id, purchasePrice:price, purchaseDate:date, addedAt:Date.now(), data:null, fetchedAt:null, error:null};
    _pos.push(pos);
    _save();
    // Clear inputs
    var inp=document.getElementById('rpf-id'); if(inp) inp.value='';
    var pinp=document.getElementById('rpf-price'); if(pinp) pinp.value='';
    _render();
    // Auto-price
    _setStatus('Pricing '+id+'…');
    var r=await apiPost('/api/realestate/price',{mode:'external-estimate',propertyId:id});
    var p=_pos.find(function(x){return x.plotId===id;});
    if(p){
      if(r.ok){ p.data=(r.d.data||r.d); p.error=null; }
      else { p.error=(r.d&&(r.d.error||r.d.detail))||'Error '+r.s; }
      p.fetchedAt=Date.now();
      _save();
    }
    _setStatus('');
    _render();
  };

  window.RE_PORTFOLIO_refreshAll = async function(){
    if(_refreshing) return;
    _refreshing=true;
    var btn=document.getElementById('rpf-refresh-btn');
    if(btn) btn.disabled=true;
    for(var i=0;i<_pos.length;i++){
      _setStatus('Refreshing '+(i+1)+'/'+_pos.length+'…');
      var r=await apiPost('/api/realestate/price',{mode:'external-estimate',propertyId:_pos[i].plotId});
      if(r.ok){ _pos[i].data=(r.d.data||r.d); _pos[i].error=null; }
      else { _pos[i].error=(r.d&&(r.d.error||r.d.detail))||'Error '+r.s; }
      _pos[i].fetchedAt=Date.now();
      _save();
      _render();
      if(i<_pos.length-1) await _sleep(700);
    }
    _setStatus('');
    if(btn) btn.disabled=false;
    _refreshing=false;
  };

  window.RE_PORTFOLIO_remove = function(plotId){
    _pos=_pos.filter(function(x){return x.plotId!==plotId;});
    _save();
    _render();
  };

  function _render(){
    var empty=document.getElementById('rpf-empty');
    var tbl  =document.getElementById('rpf-table');
    var tbody=document.getElementById('rpf-tbody');
    if(!tbody) return;

    if(!_pos.length){
      if(empty) empty.style.display='';
      if(tbl)   tbl.style.display='none';
      _updateKPI([]); _renderChart([]); return;
    }
    if(empty) empty.style.display='none';
    if(tbl)   tbl.style.display='';

    var priced=_pos.filter(function(x){return x.data&&x.data.estimate&&x.data.estimate.estimate!=null;});
    var html='';
    _pos.forEach(function(pos){
      var d=pos.data;
      var est=d&&d.estimate||{};
      var curEst=est.estimate;
      var pnl=curEst!=null&&pos.purchasePrice?curEst-pos.purchasePrice:null;
      var pnlPct=pnl!=null&&pos.purchasePrice?((pnl/pos.purchasePrice)*100):null;
      var conf=est.confidence;
      var pnlCls=pnl!=null?(pnl>0?'style="color:var(--up)"':pnl<0?'style="color:var(--dn)"':''):'';
      var band=est.estimateBandLow!=null&&est.estimateBandHigh!=null?re_fmt(est.estimateBandLow)+' – '+re_fmt(est.estimateBandHigh):'—';
      html+='<tr>';
      html+='<td style="font-weight:700;color:var(--wht)">'+_esc(pos.plotId)+'</td>';
      html+='<td style="color:var(--txt2)">'+(d&&d.propertyType||'—')+'</td>';
      html+='<td style="color:var(--txt2)">'+(d&&d.areaGroup||pos.error||'—')+'</td>';
      html+='<td class="r" style="color:var(--org)">'+(curEst!=null?re_fmtFull(curEst):'—')+'</td>';
      html+='<td class="r" style="font-size:9px;color:var(--txt3)">'+band+'</td>';
      html+='<td class="r"><span class="'+re_confClass(conf)+'">'+(conf!=null?conf+'%':'—')+'</span></td>';
      html+='<td class="r">'+(pos.purchasePrice?re_fmtFull(pos.purchasePrice):'—')+'</td>';
      html+='<td class="r" '+pnlCls+'>'+(pnl!=null?(pnl>=0?'+':'')+re_fmtFull(pnl):'—')+'</td>';
      html+='<td class="r" '+pnlCls+'>'+(pnlPct!=null?(pnlPct>=0?'+':'')+pnlPct.toFixed(1)+'%':'—')+'</td>';
      html+='<td style="font-size:9px;color:var(--txt3)">'+_esc(pos.purchaseDate||'—')+'</td>';
      html+='<td onclick="RE_PORTFOLIO_remove(\''+_esc(pos.plotId)+'\')"><span class="rpf-rm">×</span></td>';
      html+='</tr>';
    });
    tbody.innerHTML=html;
    _updateKPI(priced);
    _renderChart(priced);
  }

  function _updateKPI(priced){
    var totalVal=priced.reduce(function(s,x){return s+(x.data.estimate.estimate||0);},0);
    var totalCost=_pos.reduce(function(s,x){return s+(x.purchasePrice||0);},0);
    var pnl=priced.length&&totalCost?totalVal-totalCost:null;
    var pnlPct=pnl!=null&&totalCost?((pnl/totalCost)*100):null;
    var avgConf=priced.length?Math.round(priced.reduce(function(s,x){return s+(x.data.estimate.confidence||0);},0)/priced.length):null;

    _set('rpf-total',   priced.length?re_fmtFull(totalVal):'—');
    _set('rpf-cost',    totalCost?re_fmtFull(totalCost):'—');
    _set('rpf-count',   _pos.length);
    _set('rpf-avgconf', avgConf!=null?avgConf+'%':'—');
    var pnlEl=document.getElementById('rpf-pnl');
    var pnlPctEl=document.getElementById('rpf-pnlpct');
    if(pnlEl){
      pnlEl.textContent=pnl!=null?((pnl>=0?'+':'')+re_fmtFull(pnl)):'—';
      pnlEl.style.color=pnl!=null?(pnl>=0?'var(--up)':'var(--dn)'):'';
    }
    if(pnlPctEl){
      pnlPctEl.textContent=pnlPct!=null?((pnlPct>=0?'+':'')+pnlPct.toFixed(1)+'%'):'—';
      pnlPctEl.style.color=pnlPct!=null?(pnlPct>=0?'var(--up)':'var(--dn)'):'';
    }
  }

  function _renderChart(priced){
    var canvas=document.getElementById('rpf-chart');
    if(!canvas) return;
    _chart=dc(_chart);
    if(!priced.length) return;
    // Group by areaGroup
    var groups={};
    priced.forEach(function(x){
      var ag=(x.data&&x.data.areaGroup)||'Unknown';
      groups[ag]=(groups[ag]||0)+(x.data.estimate.estimate||0);
    });
    var labels=Object.keys(groups);
    var data=labels.map(function(l){return groups[l];});
    var COLORS=['rgba(255,140,0,.7)','rgba(0,229,255,.6)','rgba(224,64,251,.6)','rgba(0,191,165,.6)','rgba(255,214,0,.6)','rgba(0,200,83,.6)'];
    _chart=mkDoughnut(canvas, labels, data, COLORS.slice(0,labels.length));
  }

  function _setStatus(msg){ var el=document.getElementById('rpf-status'); if(el) el.textContent=msg; }
  function _set(id,v){ var el=document.getElementById(id); if(el) el.textContent=v||v===0?v:'—'; }
  function _esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/'/g,'&#39;'); }
  function _sleep(ms){ return new Promise(function(r){setTimeout(r,ms);}); }
})();
