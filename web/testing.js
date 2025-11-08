const API = '/api';
let queries = [];

function setStatus(msg){ document.getElementById('status').textContent = msg; }

async function loadQueries(){
  setStatus('Loading queries…');
  showQueriesSkeleton();
  try {
    const r = await fetch(API + '/sample-queries');
    const data = await r.json();
    queries = data.queries || [];
    const tbody = document.querySelector('#queriesTable tbody');
    tbody.innerHTML='';
    queries.forEach(q => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${q.id}</td>
        <td class='qtext'>${escapeHtml(q.text)}</td>
        <td>${q.behavioral?'<span class="flag flag-beh">Yes</span>':'No'}</td>
        <td class='answer-cell'><div id='ans-${q.id}' class='answer-snippet' aria-live='polite'></div></td>
        <td><button class='run-btn' data-id='${q.id}'>Run</button></td>`;
      tbody.appendChild(tr);
    });
    tbody.querySelectorAll('.run-btn').forEach(btn => btn.addEventListener('click', ()=>runSingle(btn.dataset.id)));
    document.getElementById('runAllBtn').disabled = false;
    setStatus('Ready');
    toast('Queries loaded','success');
  } catch(e){
    setStatus('Error');
    toast('Failed to load queries','error');
  }
}

async function runSingle(id){
  const q = queries.find(x=>x.id===id);
  if(!q) return;
  setStatus('Asking ' + id + '…');
  showAnswerSkeleton(id);
  try {
    const r = await fetch(API + '/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question: q.text})});
    const data = await r.json();
    document.getElementById('ans-'+id).textContent = data.answer;
    toast('Answer ready','success');
  } catch(e){
    document.getElementById('ans-'+id).textContent = '(error)';
    toast('Failed to get answer','error');
  } finally {
    setStatus('Ready');
  }
}

async function runAll(){
  setStatus('Running all…');
  const payload = { items: queries.map(q=>({id:q.id, question:q.text})) };
  queries.forEach(q => showAnswerSkeleton(q.id));
  try {
    const r = await fetch(API + '/bulk-ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const data = await r.json();
    let done = 0;
    data.results.forEach(res => {
      done++; 
      document.getElementById('ans-'+res.id).textContent = res.answer;
      document.getElementById('progress').textContent = done + '/' + data.results.length;
    });
    toast('All answers ready','success');
  } catch(e){
    toast('Bulk run failed','error');
  } finally {
    setStatus('Ready');
  }
}

window.addEventListener('DOMContentLoaded', ()=>{
  document.getElementById('loadBtn').addEventListener('click', loadQueries);
  document.getElementById('runAllBtn').addEventListener('click', runAll);
});

// Skeleton utilities shared with main app
function showAnswerSkeleton(id){
  const el = document.getElementById('ans-'+id);
  if(!el) return;
  el.innerHTML = `<div class='skeleton' style='height:10px;width:60%;margin-bottom:4px'></div>
                  <div class='skeleton' style='height:10px;width:72%;margin-bottom:4px'></div>
                  <div class='skeleton' style='height:10px;width:55%;'></div>`;
}
function showQueriesSkeleton(){
  const tbody = document.querySelector('#queriesTable tbody');
  tbody.innerHTML='';
  for(let i=0;i<5;i++){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class='mono'>…</td><td colspan='4'><div class='skeleton' style='height:12px;width:90%;'></div></td>`;
    tbody.appendChild(tr);
  }
}
function ensureToastContainer(){
  let c = document.querySelector('.toast-container');
  if(!c){ c = document.createElement('div'); c.className='toast-container'; document.body.appendChild(c);} return c;
}
function toast(msg,type='info',timeout=3000){
  const c = ensureToastContainer();
  const el = document.createElement('div');
  el.className = 'toast ' + type; el.textContent = msg; c.appendChild(el);
  setTimeout(()=>{ el.style.opacity='0'; el.style.transform='translateY(4px)'; setTimeout(()=> el.remove(), 350); }, timeout);
}
function escapeHtml(s){ return String(s||'').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\'':'&#39;','"':'&quot;'}[c])); }
