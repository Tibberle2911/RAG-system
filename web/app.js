const API = '/api';

async function fetchHealth(){
  try { const r = await fetch('/api/health'); return await r.json(); } catch(e){ return {status:'error'} }
}

async function ask(){
  const input = document.getElementById('question');
  const question = input.value.trim();
  if(!question){
    toast('Please enter a question first.','warn');
    return;
  }
  setStatus('Querying…');
  showAnswerSkeleton();
  try {
    const r = await fetch(API + '/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({question})});
    const data = await r.json();
    renderAnswer(data);
    toast('Answer ready','success');
  } catch(e){
    renderAnswer({answer:'(error retrieving answer)'});
    toast('Failed to retrieve answer','error');
  } finally {
    setStatus('Ready');
  }
}

function renderAnswer(data){
  const ans = document.getElementById('answer');
  ans.textContent = data.answer || '(no answer)';
}

async function search(){
  const box = document.getElementById('searchBox');
  const q = box.value.trim();
  if(!q){ toast('Enter a search term first','warn'); return; }
  setStatus('Searching…');
  showSearchSkeleton();
  try {
    const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await r.json();
    const ul = document.getElementById('searchResults');
    ul.innerHTML='';
    if(!(data.results||[]).length){
      ul.innerHTML = '<li>(no matches)</li>';
    } else {
      (data.results||[]).forEach(res => {
        const li = document.createElement('li');
        li.innerHTML = `<span class='title'>${escapeHtml(res.title)}</span><span class='score mono'>relevance ${res.score.toFixed(3)}</span><span class='content'>${escapeHtml(res.content.substring(0,220))}${res.content.length>220?'…':''}</span>`;
        ul.appendChild(li);
      });
    }
    toast('Search complete','success');
  } catch(e){
    toast('Search failed','error');
  } finally {
    setStatus('Ready');
  }
}

function setStatus(s){ document.getElementById('status').textContent = s; }

function showAnswerSkeleton(){
  const ans = document.getElementById('answer');
  ans.innerHTML = `<div class="skeleton" style="height:14px; width:65%; margin-bottom:6px"></div>
                   <div class="skeleton" style="height:14px; width:78%; margin-bottom:6px"></div>
                   <div class="skeleton" style="height:14px; width:55%;"></div>`;
}

function showSearchSkeleton(){
  const ul = document.getElementById('searchResults');
  ul.innerHTML = '';
  for(let i=0;i<4;i++){
    const li = document.createElement('li');
    li.innerHTML = `<div class='skeleton' style='height:12px;width:40%;margin-bottom:6px'></div>
                    <div class='skeleton' style='height:11px;width:85%;margin-bottom:4px'></div>
                    <div class='skeleton' style='height:11px;width:70%;'></div>`;
    ul.appendChild(li);
  }
}

function ensureToastContainer(){
  let c = document.querySelector('.toast-container');
  if(!c){
    c = document.createElement('div');
    c.className = 'toast-container';
    document.body.appendChild(c);
  }
  return c;
}

function toast(msg,type='info',timeout=3200){
  const c = ensureToastContainer();
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.setAttribute('role','alert');
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(()=>{ el.style.opacity='0'; el.style.transform='translateY(4px)'; setTimeout(()=> el.remove(), 400); }, timeout);
}

function escapeHtml(s){
  return String(s||'').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\'':'&#39;','"':'&quot;'}[c]));
}

window.addEventListener('DOMContentLoaded', async ()=>{
  document.getElementById('askBtn').addEventListener('click', ask);
  document.getElementById('searchBtn').addEventListener('click', search);
  document.getElementById('question').addEventListener('keydown', e=>{ if(e.key==='Enter'){ ask(); }});
  document.getElementById('searchBox').addEventListener('keydown', e=>{ if(e.key==='Enter'){ search(); }});
  initTabs();
  const h = await fetchHealth();
  setStatus(h.status === 'ok' ? 'Ready' : 'Service error');
});

function initTabs(){
  // Only attach tab logic to button-based tabs; anchor goes to /testing
  const buttons = document.querySelectorAll('button.tab-btn');
  buttons.forEach(btn => btn.addEventListener('click', ()=>{
    buttons.forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-content').forEach(sec => {
      sec.classList.toggle('active', sec.id === 'tab-' + target);
    });
  }));
}
