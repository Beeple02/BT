// enterprise_cms.js — Client Management Space JS
// Loaded via shell.html after CMS HTML is injected
(function(){

var _clients = [];
var _selected = null;
var _charts = {};

// ── Called by shell.html after data is fetched ────────────────────────────────
window.CMS_renderList = function(clients){
  _clients = clients || [];
  var el = document.getElementById('cms-client-list');
  if(!el) return;
  if(!_clients.length){
    el.innerHTML = '<div style="padding:20px;color:var(--txt3);font-size:10px;text-align:center">No clients yet.<br>Click + NEW to add one.</div>';
    return;
  }
  var riskColor = {conservative:'var(--up)', moderate:'var(--yel)', aggressive:'var(--dn)'};
  el.innerHTML = _clients.map(function(c){
    var rc = riskColor[c.risk_profile] || 'var(--txt2)';
    return '<div class="cms-client-item'+(c.id===(_selected&&_selected.id)?' active':'')+'" onclick="CMS_select(\''+c.id+'\')">'
      +'<div class="ci-name">'+c.name+'</div>'
      +'<div class="ci-meta" style="display:flex;gap:8px">'
      +'<span style="color:'+rc+'">'+((c.risk_profile||'—').toUpperCase())+'</span>'
      +(c.discord?'<span style="color:var(--cyn)">'+c.discord+'</span>':c.ign?'<span>'+c.ign+'</span>':'')
      +(c.onboarding_date?'<span>'+c.onboarding_date+'</span>':'')
      +'</div>'
      +'</div>';
  }).join('');
};

window.CMS_select = function(id){
  _selected = _clients.find(function(c){return c.id===id;}) || null;
  if(!_selected) return;
  // Highlight in list
  document.querySelectorAll('.cms-client-item').forEach(function(el){
    el.classList.toggle('active', el.onclick&&el.getAttribute('onclick')&&el.getAttribute('onclick').includes(id));
  });
  CMS_renderList(_clients); // re-render to update active
  CMS_renderDetail(_selected);
};

window.CMS_renderDetail = function(client){
  var main = document.getElementById('cms-main-area');
  if(!main) return;
  var empty = document.getElementById('cms-empty');
  if(empty) empty.style.display = 'none';

  var riskColor = {conservative:'var(--up)', moderate:'var(--yel)', aggressive:'var(--dn)'};
  var rc = riskColor[client.risk_profile] || 'var(--txt2)';

  main.innerHTML = '<div style="display:flex;flex-direction:column;height:100%;overflow:hidden">'
    // Detail header
    +'<div style="padding:12px 16px;background:var(--bg3);border-bottom:1px solid var(--bdr);flex-shrink:0;display:flex;align-items:center;justify-content:space-between">'
    +'<div><div style="font-size:14px;font-weight:700;color:var(--wht)">'+client.name+'</div>'
    +'<div style="font-size:9px;color:var(--txt3);margin-top:2px">'
    +'<span style="color:'+rc+'">'+((client.risk_profile||'').toUpperCase())+'</span>'
    +(client.aum_tier?' · '+(client.aum_tier||'').toUpperCase():'')
    +(client.jurisdiction?' · '+client.jurisdiction:'')
    +(client.onboarding_date?' · Since '+client.onboarding_date:'')
    +'</div></div>'
    +'<div style="display:flex;gap:8px">'
    +'<button class="btn on btn-xs" onclick="CMS_saveClient()">SAVE</button>'
    +'<button class="btn btn-xs" style="color:var(--dn);border-color:var(--dn)" onclick="CMS_deleteClient(\''+client.id+'\')">DELETE</button>'
    +'</div></div>'
    // Tabs
    +'<div class="cms-tabs">'
    +'<div class="cms-tab on" onclick="CMS_tab(\'profile\')">PROFILE</div>'
    +'<div class="cms-tab" onclick="CMS_tab(\'mandate\')">MANDATE</div>'
    +'<div class="cms-tab" onclick="CMS_tab(\'portal\')">PORTAL LINK</div>'
    +'<div class="cms-tab" onclick="CMS_tab(\'notes\')">NOTES</div>'
    +'<div class="cms-tab" onclick="CMS_tab(\'portfolios\')" id="cms-tab-portfolios">PORTFOLIOS</div>'
    +'<div class="cms-tab" onclick="CMS_tab(\'documents\')">DOCUMENTS</div>'
    +'<div class="cms-tab" onclick="CMS_tab(\'report\')">REPORT SCHED</div>'
    +'</div>'
    // Profile panel
    +'<div class="cms-panel on scroll" id="cms-panel-profile">'
    +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'
    +'<div class="field-row"><label>CLIENT NAME</label><input id="edit-name" value="'+(client.name||'')+'"></div>'
    +'<div class="field-row"><label>DISCORD @</label><input id="edit-discord" value="'+(client.discord||'')+'"></div>'
    +'<div class="field-row"><label>IGN</label><input id="edit-ign" value="'+(client.ign||'')+'"></div>'
    +'<div class="field-row"><label>RISK PROFILE</label><select id="edit-risk">'
    +'<option value="conservative"'+(client.risk_profile==='conservative'?' selected':'')+'>Conservative</option>'
    +'<option value="moderate"'+(client.risk_profile==='moderate'?' selected':'')+'>Moderate</option>'
    +'<option value="aggressive"'+(client.risk_profile==='aggressive'?' selected':'')+'>Aggressive</option>'
    +'</select></div>'
    +'<div class="field-row"><label>AUM TIER</label><select id="edit-tier">'
    +'<option value="retail"'+(client.aum_tier==='retail'?' selected':'')+'>Retail (&lt;$100K)</option>'
    +'<option value="hnw"'+(client.aum_tier==='hnw'?' selected':'')+'>HNW ($100K–$1M)</option>'
    +'<option value="uhnw"'+(client.aum_tier==='uhnw'?' selected':'')+'>UHNW (&gt;$1M)</option>'
    +'<option value="institutional"'+(client.aum_tier==='institutional'?' selected':'')+'>Institutional</option>'
    +'</select></div>'
    +'<div class="field-row"><label>ONBOARDING DATE</label><input id="edit-onboard" type="date" value="'+(client.onboarding_date||'')+'"></div>'
    +'<div class="field-row"><label>JURISDICTION</label><input id="edit-jurisdiction" placeholder="e.g. Redmont, Alexandria" value="'+(client.jurisdiction||'')+'"></div>'
    +'<div class="field-row" style="grid-column:1/-1"><label>OVERVIEW NOTES</label><textarea id="edit-notes" rows="3">'+(client.notes||'')+'</textarea></div>'
    +'</div></div>'
    // Mandate panel
    +'<div class="cms-panel scroll" id="cms-panel-mandate">'
    +'<div style="font-size:9px;color:var(--txt3);margin-bottom:12px">Define investment constraints for this client. Saved with the client profile.</div>'
    +'<div class="mandate-row"><label>Max single position (%)</label><input type="number" id="m-max-pos" value="'+(client.mandate_max_pos||40)+'" min="1" max="100"></div>'
    +'<div class="mandate-row"><label>Allowed exchanges</label>'
    +'<label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="m-ner" '+(client.mandate_ner!==false?'checked':'')+'>NER</label>'
    +'<label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="m-tse" '+(client.mandate_tse!==false?'checked':'')+'>TSE</label>'
    +'</div>'
    +'<div class="mandate-row"><label>Excluded tickers (comma-separated)</label><input type="text" id="m-excluded" value="'+(client.mandate_excluded||'')+'" style="width:200px" placeholder="e.g. BB,RTG"></div>'
    +'<div class="mandate-row"><label>Min cash buffer (%)</label><input type="number" id="m-min-cash" value="'+(client.mandate_min_cash||5)+'" min="0" max="100"></div>'
    +'<div class="mandate-row"><label>Max leverage (1 = no leverage)</label><input type="number" id="m-max-lev" value="'+(client.mandate_max_lev||1)+'" min="1" max="10" step="0.1"></div>'
    +'<div style="margin-top:12px"><button class="btn on btn-xs" onclick="CMS_saveMandate()">SAVE MANDATE</button></div>'
    +'</div>'
    // Portal panel
    +'<div class="cms-panel scroll" id="cms-panel-portal">'
    +'<div style="font-size:10px;color:var(--txt2);margin-bottom:12px;line-height:1.7">'
    +'A portal link gives your client a <b>read-only view</b> of their portfolio — no login required.<br>'
    +'It automatically shows all portfolios where the client name matches <b>'+client.name+'</b>.</div>'
    +(client.portal_token?'<div style="font-size:9px;color:var(--txt3);margin-bottom:6px">CURRENT PORTAL LINK</div><div class="portal-link-box" id="portal-url-display">'+window.location.origin+'/portal/'+client.portal_token+'</div><div style="display:flex;gap:8px;margin-top:8px"><button class="btn btn-xs" onclick="CMS_copyPortal()">COPY LINK</button><button class="btn btn-xs" onclick="window.open(\''+window.location.origin+'/portal/'+client.portal_token+'\',\'_blank\')">OPEN ↗</button><button class="btn btn-xs" style="color:var(--org);border-color:var(--org)" onclick="CMS_genPortal(\''+client.id+'\')">REGENERATE</button></div>'
    :'<button class="btn on btn-xs" onclick="CMS_genPortal(\''+client.id+'\')">GENERATE PORTAL LINK →</button>')
    +'<div id="portal-gen-result" style="margin-top:10px"></div>'
    +'</div>'
    // Notes panel
    +'<div class="cms-panel scroll" id="cms-panel-notes">'
    +'<div style="display:flex;gap:8px;margin-bottom:12px;align-items:flex-end">'
    +'<div class="field-row" style="flex:1;margin-bottom:0"><label>NEW NOTE</label><textarea id="note-text" rows="2" placeholder="Meeting notes, follow-ups, decisions…"></textarea></div>'
    +'<button class="btn on btn-xs" onclick="CMS_addNote(\''+client.id+'\')">ADD</button>'
    +'</div>'
    +'<div id="notes-log">'
    +(((client.notes_log||[]).slice().reverse()).map(function(n){
      return '<div class="note-item"><div class="note-ts"><span class="note-author">'+(n.author||'PM')+'</span>'+new Date(n.ts).toLocaleString('en-GB')+'</div><div>'+n.text+'</div></div>';
    }).join('')||'<div style="color:var(--txt3);font-size:10px">No notes yet.</div>')
    +'</div></div>'
    // Portfolios panel
    +'<div class="cms-panel scroll" id="cms-panel-portfolios">'
    +'<div style="font-size:9px;color:var(--txt3);margin-bottom:12px">Link portfolios to this client. Linked portfolios appear in their portal and firm analytics.</div>'
    +'<div id="cms-pf-linker"><div style="color:var(--txt3);font-size:10px">Loading portfolios…</div></div>'
    +'</div>'
    // Documents panel
    +'<div class="cms-panel scroll" id="cms-panel-documents">'
    +'<div style="font-size:9px;color:var(--txt3);margin-bottom:12px">Store links to mandate PDFs, signed agreements, reports, and other documents. No file upload — just links to where the doc already lives (Drive, Dropbox, internal URL).</div>'
    +'<div style="display:grid;grid-template-columns:1fr 1fr 120px 1fr;gap:6px;align-items:end;margin-bottom:10px">'
    +'<div class="field-row" style="margin:0"><label>NAME</label><input id="doc-name" placeholder="e.g. Mandate 2025"></div>'
    +'<div class="field-row" style="margin:0"><label>TYPE</label><select id="doc-type"><option value="Mandate">Mandate</option><option value="Agreement">Agreement</option><option value="Report">Report</option><option value="Other">Other</option></select></div>'
    +'<div class="field-row" style="margin:0"><label>DATE</label><input id="doc-date" type="date"></div>'
    +'<div class="field-row" style="margin:0"><label>URL / LINK</label><input id="doc-url" placeholder="https://…"></div>'
    +'</div>'
    +'<div style="margin-bottom:14px"><button class="btn on btn-xs" onclick="CMS_addDoc(\''+client.id+'\')">+ ADD DOCUMENT</button></div>'
    +'<div id="cms-doc-list">'
    +_renderDocList(client.documents||[],client.id)
    +'</div></div>'
    // Report Schedule panel
    +'<div class="cms-panel scroll" id="cms-panel-report">'
    +'<div style="font-size:9px;color:var(--txt3);margin-bottom:14px;line-height:1.8">Configure automatic report generation for this client. When enabled, a tearsheet snapshot is flagged for generation on the chosen schedule.</div>'
    +'<div class="mandate-row"><label>SCHEDULE ENABLED</label><input type="checkbox" id="rs-enabled" '+(client.report_schedule&&client.report_schedule.enabled?'checked':'')+' style="accent-color:var(--org)"></div>'
    +'<div class="mandate-row"><label>FREQUENCY</label><select id="rs-freq" style="width:120px;background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:3px 6px">'
    +'<option value="weekly"'+(client.report_schedule&&client.report_schedule.frequency==='weekly'?' selected':'')+'>Weekly</option>'
    +'<option value="monthly"'+((!client.report_schedule||client.report_schedule.frequency==='monthly')?' selected':'')+'>Monthly</option>'
    +'<option value="quarterly"'+(client.report_schedule&&client.report_schedule.frequency==='quarterly'?' selected':'')+'>Quarterly</option>'
    +'</select></div>'
    +'<div class="mandate-row"><label>FORMAT</label><select id="rs-format" style="width:120px;background:var(--bg);border:1px solid var(--bdr2);color:var(--wht);font-family:var(--font);font-size:10px;padding:3px 6px">'
    +'<option value="PDF"'+((!client.report_schedule||client.report_schedule.format==='PDF')?' selected':'')+'>PDF Tearsheet</option>'
    +'<option value="CSV"'+(client.report_schedule&&client.report_schedule.format==='CSV'?' selected':'')+'>CSV Summary</option>'
    +'</select></div>'
    +(client.report_schedule&&client.report_schedule.last_generated
      ?'<div class="mandate-row"><label>LAST GENERATED</label><span style="color:var(--up);font-size:10px">'+client.report_schedule.last_generated+'</span></div>'
      +'<div class="mandate-row"><label>NEXT DUE</label><span style="color:var(--yel);font-size:10px">'+_calcNextReport(client.report_schedule)+'</span></div>'
      :'<div style="margin-top:10px;font-size:9px;color:var(--txt3)">No reports generated yet.</div>')
    +'<div style="margin-top:16px;display:flex;gap:8px">'
    +'<button class="btn on btn-xs" onclick="CMS_saveReportSchedule(\''+client.id+'\')">SAVE SCHEDULE</button>'
    +'<button class="btn btn-xs" onclick="CMS_generateNow(\''+client.id+'\')">⚡ GENERATE NOW</button>'
    +'</div>'
    +'<div id="rs-status" style="margin-top:8px;font-size:9px"></div>'
    +'</div>'
    +'</div>';
};

window.CMS_tab = function(tab){
  var tabNames = ['profile','mandate','portal','notes','portfolios','documents','report'];
  document.querySelectorAll('.cms-tab').forEach(function(t,i){
    t.classList.toggle('on', tabNames[i]===tab);
  });
  document.querySelectorAll('.cms-panel').forEach(function(p){
    p.classList.toggle('on', p.id==='cms-panel-'+tab);
  });
  if(tab==='portfolios' && _selected) CMS_loadPfLinker(_selected.id);
};

window.CMS_saveClient = async function(){
  if(!_selected) return;
  var body = Object.assign({}, _selected, {
    name:            (document.getElementById('edit-name')||{}).value || _selected.name,
    discord:         (document.getElementById('edit-discord')||{}).value||'',
    ign:             (document.getElementById('edit-ign')||{}).value||'',
    risk_profile:    (document.getElementById('edit-risk')||{}).value||'moderate',
    aum_tier:        (document.getElementById('edit-tier')||{}).value||'uhnw',
    onboarding_date: (document.getElementById('edit-onboard')||{}).value||'',
    jurisdiction:    (document.getElementById('edit-jurisdiction')||{}).value||'',
    notes:           (document.getElementById('edit-notes')||{}).value||'',
  });
  var r = await apiPost('/api/enterprise/clients', body);
  if(!r.ok){ alert('Error saving'); return; }
  _selected = r.d;
  _clients = _clients.map(function(c){return c.id===r.d.id?r.d:c;});
  CMS_renderList(_clients);
  var bar = document.getElementById('cmd-st');
  if(bar){bar.textContent='Client saved: '+r.d.name;bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);}
};

window.CMS_saveMandate = async function(){
  if(!_selected) return;
  var body = Object.assign({}, _selected, {
    mandate_max_pos:  parseFloat((document.getElementById('m-max-pos')||{}).value)||40,
    mandate_ner:      (document.getElementById('m-ner')||{}).checked!==false,
    mandate_tse:      (document.getElementById('m-tse')||{}).checked!==false,
    mandate_excluded: (document.getElementById('m-excluded')||{}).value||'',
    mandate_min_cash: parseFloat((document.getElementById('m-min-cash')||{}).value)||5,
    mandate_max_lev:  parseFloat((document.getElementById('m-max-lev')||{}).value)||1,
  });
  var r = await apiPost('/api/enterprise/clients', body);
  if(!r.ok){ alert('Error saving mandate'); return; }
  _selected = r.d;
  _clients = _clients.map(function(c){return c.id===r.d.id?r.d:c;});
  var bar=document.getElementById('cmd-st');
  if(bar){bar.textContent='Mandate saved';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},2000);}
};

window.CMS_genPortal = async function(cid){
  var r = await apiPost('/api/enterprise/clients/'+cid+'/portal_token', {});
  if(!r.ok){ alert('Error generating portal link'); return; }
  var url = r.d.url;
  var el = document.getElementById('portal-gen-result');
  if(el) el.innerHTML = '<div style="font-size:9px;color:var(--txt3);margin-bottom:6px">PORTAL LINK GENERATED</div>'
    +'<div class="portal-link-box">'+url+'</div>'
    +'<div style="display:flex;gap:8px;margin-top:8px">'
    +'<button class="btn btn-xs" onclick="navigator.clipboard.writeText(\''+url+'\')">COPY</button>'
    +'<button class="btn btn-xs" onclick="window.open(\''+url+'\',\'_blank\')">OPEN ↗</button>'
    +'</div>';
  // Update client
  _selected = Object.assign({}, _selected, {portal_token: r.d.token});
  _clients = _clients.map(function(c){return c.id===_selected.id?_selected:c;});
};

window.CMS_copyPortal = function(){
  var el = document.getElementById('portal-url-display');
  if(el) navigator.clipboard.writeText(el.textContent.trim());
  var bar=document.getElementById('cmd-st');
  if(bar){bar.textContent='Portal link copied';bar.style.color='var(--up)';setTimeout(function(){bar.textContent='ENTERPRISE SPACE';bar.style.color='var(--txt3)';},1500);}
};

window.CMS_addNote = async function(cid){
  var text = (document.getElementById('note-text')||{}).value||'';
  if(!text.trim()) return;
  var r = await apiPost('/api/enterprise/clients/'+cid+'/note', {text:text});
  if(!r.ok){ alert('Error saving note'); return; }
  document.getElementById('note-text').value = '';
  // Refresh
  var fresh = await api('/api/enterprise/clients/'+cid);
  if(fresh.ok){
    _selected = fresh.d;
    _clients  = _clients.map(function(c){return c.id===_selected.id?_selected:c;});
    CMS_renderDetail(_selected);
    CMS_tab('notes');
  }
};

window.CMS_deleteClient = async function(cid){
  if(!confirm('Delete this client? Cannot be undone.')) return;
  await api('/api/enterprise/clients/'+cid, {method:'DELETE'});
  _clients = _clients.filter(function(c){return c.id!==cid;});
  _selected = null;
  CMS_renderList(_clients);
  var main = document.getElementById('cms-main-area');
  if(main) main.innerHTML = '<div style="flex:1;display:flex;align-items:center;justify-content:center;color:var(--txt3);font-size:11px">Select a client or create a new one.</div>';
};

window.CMS_newClient  = function(){ var m=document.getElementById('cms-new-modal'); if(m) m.style.display='flex'; };
window.CMS_hideNewModal = function(){ var m=document.getElementById('cms-new-modal'); if(m) m.style.display='none'; };

window.CMS_createClient = async function(){
  var name = (document.getElementById('nc-name')||{}).value||'';
  if(!name.trim()){ alert('Client name required'); return; }
  var body = {
    name: name,
    discord:         (document.getElementById('nc-discord')||{}).value||'',
    ign:             (document.getElementById('nc-ign')||{}).value||'',
    risk_profile:    (document.getElementById('nc-risk')||{}).value||'moderate',
    aum_tier:        (document.getElementById('nc-tier')||{}).value||'uhnw',
    onboarding_date: (document.getElementById('nc-onboard')||{}).value||'',
    jurisdiction:    (document.getElementById('nc-jurisdiction')||{}).value||'',
    notes:           (document.getElementById('nc-notes')||{}).value||'',
  };
  var r = await apiPost('/api/enterprise/clients', body);
  if(!r.ok){ alert('Error creating client'); return; }
  CMS_hideNewModal();
  ['nc-name','nc-email','nc-phone','nc-onboard','nc-jurisdiction','nc-notes'].forEach(function(id){var el=document.getElementById(id);if(el)el.value='';});
  _clients.push(r.d);
  CMS_renderList(_clients);
  CMS_select(r.d.id);
};

window.CMS_loadPfLinker = async function(cid){
  var el = document.getElementById('cms-pf-linker');
  if(!el) return;

  // Fetch all portfolios + current links in parallel
  var rAll  = await api('/api/enterprise/portfolios');
  var rLinked = await api('/api/enterprise/clients/'+cid+'/portfolios');
  if(!rAll.ok){ el.innerHTML='<div style="color:var(--dn);font-size:10px">Failed to load portfolios.</div>'; return; }

  var allPfs  = rAll.d || [];
  var linkedIds = (rLinked.ok ? rLinked.d.linked_pf_ids : []) || [];

  if(!allPfs.length){
    el.innerHTML='<div style="color:var(--txt3);font-size:10px">No portfolios exist yet. Create one in the PORTFOLIO space first.</div>';
    return;
  }

  el.innerHTML = '<table class="dt" style="width:100%;margin-bottom:12px">'
    +'<thead><tr><th style="width:32px"></th><th>PORTFOLIO</th><th>CLIENT FIELD</th><th class="r">POSITIONS</th></tr></thead>'
    +'<tbody>'
    +allPfs.map(function(pf){
      var checked = linkedIds.indexOf(pf.id) !== -1;
      var clientMatch = (pf.client||'').trim() ? '' : '<span style="font-size:8px;color:var(--yel)"> (no client set)</span>';
      return '<tr>'
        +'<td><input type="checkbox" data-pfid="'+pf.id+'" '+(checked?'checked':'')+' style="accent-color:var(--org);cursor:pointer"></td>'
        +'<td style="color:var(--wht);font-weight:600">'+pf.name+'</td>'
        +'<td style="color:var(--txt2)">'+(pf.client||'—')+clientMatch+'</td>'
        +'<td class="r" style="color:var(--txt2)">'+(Array.isArray(pf.positions)?pf.positions.length:(pf.positions||0))+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table>'
    +'<button class="btn on btn-xs" onclick="CMS_savePfLinks(&quot;'+cid+'&quot;)">SAVE LINKS →</button>'
    +'<div id="cms-pf-link-status" style="display:inline-block;margin-left:10px;font-size:9px"></div>';
};

// ── Document helpers ──────────────────────────────────────────────────────────
function _renderDocList(docs, cid){
  if(!docs||!docs.length) return '<div style="color:var(--txt3);font-size:10px">No documents yet.</div>';
  return '<table class="dt" style="width:100%"><thead><tr><th>NAME</th><th>TYPE</th><th>DATE</th><th>LINK</th><th></th></tr></thead><tbody>'
    +docs.map(function(d){
      var typeC = {Mandate:'var(--org)',Agreement:'var(--cyn)',Report:'var(--up)',Other:'var(--txt2)'}[d.type]||'var(--txt2)';
      return '<tr>'
        +'<td style="color:var(--wht);font-weight:600">'+d.name+'</td>'
        +'<td><span style="color:'+typeC+';font-size:9px">'+d.type+'</span></td>'
        +'<td style="color:var(--txt2)">'+d.date+'</td>'
        +'<td>'+(d.url?'<a href="'+d.url+'" target="_blank" style="color:var(--cyn);font-size:9px;text-decoration:none">OPEN ↗</a>':'—')+'</td>'
        +'<td><span style="color:var(--dn);cursor:pointer;font-size:10px" onclick="CMS_removeDoc(\''+cid+'\',\''+d.id+'\')">✕</span></td>'
        +'</tr>';
    }).join('')+'</tbody></table>';
}

window.CMS_addDoc = async function(cid){
  var name = (document.getElementById('doc-name')||{}).value||'';
  var url  = (document.getElementById('doc-url')||{}).value||'';
  if(!name.trim()){ alert('Document name required'); return; }
  var doc = {
    id: Date.now().toString(36),
    name: name,
    type: (document.getElementById('doc-type')||{}).value||'Other',
    date: (document.getElementById('doc-date')||{}).value||new Date().toISOString().slice(0,10),
    url:  url,
  };
  var docs = (_selected.documents||[]).concat([doc]);
  var body = Object.assign({}, _selected, {documents: docs});
  var r = await apiPost('/api/enterprise/clients', body);
  if(!r.ok){ alert('Error saving document'); return; }
  _selected = r.d;
  _clients  = _clients.map(function(c){return c.id===r.d.id?r.d:c;});
  var el = document.getElementById('cms-doc-list');
  if(el) el.innerHTML = _renderDocList(_selected.documents||[], cid);
  ['doc-name','doc-url','doc-date'].forEach(function(id){var e=document.getElementById(id);if(e)e.value='';});
};

window.CMS_removeDoc = async function(cid, docId){
  if(!_selected) return;
  var docs = (_selected.documents||[]).filter(function(d){return d.id!==docId;});
  var body = Object.assign({}, _selected, {documents: docs});
  var r = await apiPost('/api/enterprise/clients', body);
  if(!r.ok){ alert('Error removing document'); return; }
  _selected = r.d;
  _clients  = _clients.map(function(c){return c.id===r.d.id?r.d:c;});
  var el = document.getElementById('cms-doc-list');
  if(el) el.innerHTML = _renderDocList(_selected.documents||[], cid);
};

// ── Report Schedule helpers ───────────────────────────────────────────────────
function _calcNextReport(sched){
  if(!sched||!sched.last_generated) return 'Pending first run';
  var d = new Date(sched.last_generated);
  if(sched.frequency==='weekly')    d.setDate(d.getDate()+7);
  else if(sched.frequency==='monthly') d.setMonth(d.getMonth()+1);
  else d.setMonth(d.getMonth()+3);
  return d.toISOString().slice(0,10);
}

window.CMS_saveReportSchedule = async function(cid){
  if(!_selected) return;
  var sched = {
    enabled:   (document.getElementById('rs-enabled')||{}).checked||false,
    frequency: (document.getElementById('rs-freq')||{}).value||'monthly',
    format:    (document.getElementById('rs-format')||{}).value||'PDF',
    last_generated: _selected.report_schedule ? _selected.report_schedule.last_generated : null,
  };
  var body = Object.assign({}, _selected, {report_schedule: sched});
  var r = await apiPost('/api/enterprise/clients', body);
  var st = document.getElementById('rs-status');
  if(r.ok){
    _selected = r.d; _clients = _clients.map(function(c){return c.id===r.d.id?r.d:c;});
    if(st){st.textContent='✓ Schedule saved';st.style.color='var(--up)';setTimeout(function(){st.textContent='';},2000);}
  } else {
    if(st){st.textContent='Error saving';st.style.color='var(--dn)';}
  }
};

window.CMS_generateNow = async function(cid){
  if(!_selected) return;
  var now = new Date().toISOString().slice(0,10);
  var sched = Object.assign({}, _selected.report_schedule||{enabled:false,frequency:'monthly',format:'PDF'}, {last_generated: now});
  var body  = Object.assign({}, _selected, {report_schedule: sched});
  var r = await apiPost('/api/enterprise/clients', body);
  var st = document.getElementById('rs-status');
  if(r.ok){
    _selected = r.d; _clients = _clients.map(function(c){return c.id===r.d.id?r.d:c;});
    if(st){st.textContent='✓ Report generation flagged — last generated: '+now;st.style.color='var(--up)';}
  } else {
    if(st){st.textContent='Error';st.style.color='var(--dn)';}
  }
};

window.CMS_savePfLinks = async function(cid){
  var checked = [];
  document.querySelectorAll('#cms-pf-linker input[type=checkbox]').forEach(function(cb){
    if(cb.checked) checked.push(cb.dataset.pfid);
  });
  var r = await apiPost('/api/enterprise/clients/'+cid+'/portfolios', {pf_ids: checked});
  var st = document.getElementById('cms-pf-link-status');
  if(r.ok){
    if(st){ st.textContent = '✓ '+checked.length+' portfolio(s) linked'; st.style.color='var(--up)'; }
    // Update local client object
    if(_selected) _selected.linked_pf_ids = checked;
  } else {
    if(st){ st.textContent = 'Error saving'; st.style.color='var(--dn)'; }
  }
};

})();
