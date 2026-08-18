/* Sentinel dashboard client.
   No framework and no build step on purpose: the page is a few hundred lines of
   DOM writing, and a toolchain here would be more moving parts than the thing it
   renders. Polling intervals are staggered so a sleeping free-tier container
   wakes up to one request at a time rather than five. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmt = (n) => (n === null || n === undefined ? '—' : n.toLocaleString());
const clock = (ts) => new Date(ts * 1000).toLocaleTimeString(undefined, { hour12: false });

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

/* ---------------------------------------------------------------- health */

async function refreshHealth() {
  let h;
  try {
    h = await api('/api/health');
  } catch {
    $('#status-dot').className = 'dot bad';
    $('#status-text').textContent = 'unreachable';
    return;
  }
  const dot = $('#status-dot');
  dot.className = 'dot ' + (h.status === 'ok' ? 'live' : 'warn');
  $('#status-text').textContent =
    h.status === 'ok' ? `${h.sources_healthy}/${h.sources_total} sources healthy` : 'degraded';

  const up = h.uptime_s || 0;
  $('#pill-uptime').textContent =
    up > 3600 ? `uptime ${(up / 3600).toFixed(1)}h` : `uptime ${Math.round(up / 60)}m`;

  $('#s-jobs').textContent = fmt(h.jobs.total);
  $('#s-jobs-n').textContent = `${fmt(h.jobs.duplicates)} flagged as near-duplicate`;
  $('#s-new').textContent = fmt(h.jobs.new_24h);
  $('#s-clean').textContent = h.last_hour.requests ? `${h.last_hour.clean_pct}%` : '—';
  $('#s-req').textContent = `${fmt(h.last_hour.requests)} requests`;
  $('#s-src').textContent = `${h.sources_healthy}/${h.sources_total}`;
}

/* --------------------------------------------------------------- sources */

const BREAKER_CLASS = { closed: 'ok', half_open: 'warn', open: 'bad' };

function sourceCard(s) {
  const card = el('div', `card state-${s.breaker}${s.quarantined ? ' quarantined' : ''}`);

  const head = el('div', 'card-head');
  head.append(el('span', 'name', s.name));
  head.append(el('span', 'badge', s.kind));
  if (s.synthetic) head.append(el('span', 'badge synthetic', 'synthetic data'));
  head.append(el('span', `badge ${BREAKER_CLASS[s.breaker] || ''}`, `circuit ${s.breaker}`));
  if (s.quarantined) head.append(el('span', 'badge warn', 'quarantined'));
  if (s.excluded_by_robots) head.append(el('span', 'badge warn', 'excluded by robots.txt'));
  card.append(head);

  // Strategy ladder: which rung we are on, and which ones we have fallen past.
  const ladder = el('div', 'ladder');
  const activeIdx = s.strategies.findIndex((x) => x.name === s.current_strategy);
  s.strategies.forEach((st, i) => {
    const cls = i === activeIdx ? 'rung active' : i < activeIdx ? 'rung spent' : 'rung';
    ladder.append(el('span', cls, st.name));
  });
  card.append(ladder);

  const kv = el('dl', 'kv');
  const pair = (k, v) => { kv.append(el('dt', null, k)); kv.append(el('dd', null, v)); };
  pair('identity', s.identity.identity || '—');
  pair('jobs', `${fmt(s.job_count)}`);
  pair('recent ok', s.success_rate_recent === null ? '—' : `${Math.round(s.success_rate_recent * 100)}%`);
  pair('next poll', `${Math.round(s.next_due_in_s)}s`);
  if (s.note) pair('last note', s.note.slice(0, 90));
  card.append(kv);

  // Thompson posteriors, one bar per pacing tier. The bar is the posterior mean
  // success rate, so you can watch the bandit change its mind under pressure.
  const tiers = Object.keys(s.pace_posterior || {});
  if (tiers.length) {
    const bars = el('div', 'bandit');
    const labels = el('div', 'bandit-labels');
    tiers.forEach((t) => {
      const b = el('div', 'b');
      const fill = el('span');
      fill.style.height = `${Math.round(s.pace_posterior[t].mean * 100)}%`;
      b.append(fill);
      b.title = `${t}: mean ${s.pace_posterior[t].mean}, n=${s.pace_posterior[t].n}`;
      bars.append(b);
      labels.append(el('div', null, t.slice(0, 4)));
    });
    card.append(bars);
    card.append(labels);
  }

  const actions = el('div', 'card-actions');
  const poll = el('button', 'btn primary', 'poll now');
  poll.onclick = async () => {
    poll.disabled = true;
    poll.textContent = 'polling…';
    try { await api(`/api/sources/${s.name}/poll`, { method: 'POST' }); }
    finally { poll.disabled = false; poll.textContent = 'poll now'; refreshSources(); refreshHealth(); }
  };
  const reset = el('button', 'btn', 'reset circuit');
  reset.onclick = async () => {
    await api(`/api/sources/${s.name}/reset`, { method: 'POST' });
    refreshSources();
  };
  actions.append(poll, reset);
  card.append(actions);

  const lic = el('div', 'licence');
  lic.append(document.createTextNode(s.licence_note + ' '));
  if (s.homepage) {
    const a = el('a', null, 'source ↗');
    a.href = s.homepage;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    lic.append(a);
  }
  card.append(lic);
  return card;
}

let sourceNames = [];

async function refreshSources() {
  let rows;
  try { rows = await api('/api/sources'); } catch { return; }
  const box = $('#sources');
  box.textContent = '';
  rows.forEach((s) => box.append(sourceCard(s)));

  if (sourceNames.length !== rows.length) {
    sourceNames = rows.map((r) => r.name);
    const sel = $('#f-source');
    sel.textContent = '';
    sel.append(new Option('all sources', ''));
    sourceNames.forEach((n) => sel.append(new Option(n, n)));
  }
}

/* --------------------------------------------------------------- sandbox */

const DEFENCES = [
  ['fingerprint_check', 'fingerprint check',
   'Rejects incoherent header sets — a Chrome UA with no client hints, a library UA, a bare */* Accept.'],
  ['rate_limit', 'rate limit',
   'Sliding window per client. Returns 429 with Retry-After once the budget is spent.'],
  ['captcha_wall', 'captcha wall',
   'HTTP 200 with a challenge body. The soft block a status-code check cannot see.'],
  ['hard_block', 'hard block',
   'Unconditional 403 with a vendor-style block page.'],
  ['silent_empty', 'silent empty',
   'Correct status, correct content-type, zero rows. Fails silently by design.'],
];

function buildToggles(state) {
  const box = $('#toggles');
  box.textContent = '';
  DEFENCES.forEach(([key, label, desc]) => {
    const wrap = el('label', 'toggle' + (state[key] ? ' on' : ''));
    const cb = el('input');
    cb.type = 'checkbox';
    cb.checked = !!state[key];
    cb.onchange = async () => {
      await api('/sandbox/control', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ [key]: cb.checked }),
      });
      refreshSandbox();
    };
    const txt = el('div');
    txt.append(el('div', 't', label));
    txt.append(el('div', 'd', desc));
    wrap.append(cb, txt);
    box.append(wrap);
  });
}

async function refreshSandbox() {
  let s;
  try { s = await api('/sandbox/state'); } catch { $('#sandbox-section').style.display = 'none'; return; }
  buildToggles(s);
  $('#markup').value = String(s.markup_version);
  $('#sandbox-counters').textContent = `${fmt(s.hits)} hits · ${fmt(s.blocked)} blocked`;
}

/* ----------------------------------------------------------------- feed */

function pushEvent(ev, prepend = true) {
  const feed = $('#feed');
  const row = el('div', `ev ${ev.level}`);
  row.append(el('span', 't', clock(ev.ts)));
  row.append(el('span', 's', `${ev.source || '—'}/${ev.kind}`));
  row.append(el('span', 'm', ev.message));
  if (prepend) feed.prepend(row); else feed.append(row);
  while (feed.children.length > 120) feed.lastChild.remove();
}

async function startFeed() {
  try {
    const seed = await api('/api/events?limit=40');
    seed.forEach((e) => pushEvent(e, false));
  } catch { /* the stream below will fill it in */ }

  const es = new EventSource('/api/events/stream');
  es.onmessage = (m) => {
    try { pushEvent(JSON.parse(m.data)); } catch { /* keepalive */ }
  };
  // EventSource reconnects on its own; the only thing worth doing on error is
  // not spamming the console about a free host that went to sleep.
  es.onerror = () => {};
}

/* ----------------------------------------------------------------- jobs */

let jobsTimer = null;

async function refreshJobs() {
  const params = new URLSearchParams({
    q: $('#q').value.trim(),
    source: $('#f-source').value,
    role: $('#f-role').value,
    seniority: $('#f-seniority').value,
    limit: '30',
  });
  let data;
  try { data = await api('/api/jobs?' + params); } catch { return; }

  const box = $('#jobs');
  box.textContent = '';
  if (!data.results.length) {
    box.append(el('div', 'empty',
      data.query ? `Nothing close to “${data.query}” yet — the index rebuilds as jobs land.`
                 : 'No jobs stored yet. The first poll lands within a minute of boot.'));
    return;
  }

  data.results.forEach((j) => {
    const card = el('div', 'job');
    const title = el('div', 'jt');
    if (j.score !== undefined) {
      const sc = el('span', 'score', j.score.toFixed(3));
      title.append(sc);
    }
    const link = el('a', null, j.title);
    link.href = j.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    title.append(link);
    card.append(title);
    card.append(el('div', 'jm',
      [j.company || 'company not stated', j.location || 'location not stated']
        .filter(Boolean).join(' · ')));

    const chips = el('div', 'chips');
    chips.append(el('span', 'badge', j.source));
    if (j.synthetic) chips.append(el('span', 'badge synthetic', 'synthetic'));
    if (j.role_family) chips.append(el('span', 'badge', j.role_family));
    if (j.seniority) chips.append(el('span', 'badge', j.seniority));
    if (j.remote) chips.append(el('span', 'badge ok', 'remote'));
    if (j.salary_text) chips.append(el('span', 'badge', j.salary_text));
    if (j.confidence < 0.9) {
      chips.append(el('span', 'badge warn', `extraction ${Math.round(j.confidence * 100)}%`));
    }
    card.append(chips);
    box.append(card);
  });
}

function debounceJobs() {
  clearTimeout(jobsTimer);
  jobsTimer = setTimeout(refreshJobs, 260);
}

/* ------------------------------------------------------------------- ml */

function panelTitle(node, title) {
  node.textContent = '';
  const head = el('div', 'card-head');
  head.append(el('span', 'name', title));
  node.append(head);
}

async function refreshML() {
  let m;
  try { m = await api('/api/ml'); } catch { return; }

  // Block classifier
  const b = $('#ml-block');
  panelTitle(b, 'Soft-block classifier');
  const bc = m.block_classifier || {};
  const kv = el('dl', 'kv');
  const pair = (k, v) => { kv.append(el('dt', null, k)); kv.append(el('dd', null, v)); };
  pair('model', 'logistic regression');
  pair('precision', bc.precision ?? '—');
  pair('recall', bc.recall ?? '—');
  pair('avg precision', bc.avg_precision ?? '—');
  pair('held out', `${bc.n_holdout ?? '—'} samples`);
  b.append(kv);

  if (bc.top_weights) {
    const w = el('div', 'weights');
    const max = Math.max(...Object.values(bc.top_weights).map(Math.abs), 1);
    Object.entries(bc.top_weights).forEach(([name, val]) => {
      const row = el('div', 'w');
      row.append(el('div', null, name));
      const bar = el('div', 'bar');
      const i = el('i', val > 0 ? 'pos' : 'neg');
      i.style.width = `${Math.round((Math.abs(val) / max) * 100)}%`;
      bar.append(i);
      row.append(bar);
      row.append(el('div', 'n', val.toFixed(2)));
      w.append(row);
    });
    b.append(w);
  }
  // The live confusion matrix sits next to the holdout metrics on purpose:
  // one is measured on the model's own training distribution, the other is
  // measured on real traffic against ground truth. Showing only the flattering
  // one would be the exact trap this project keeps warning about.
  const live = m.block_classifier_live || {};
  const liveBox = el('div', 'licence');
  if (live.n) {
    liveBox.append(el('div', 't',
      `Live: ${live.true_positive} TP · ${live.false_positive} FP · ` +
      `${live.true_negative} TN · ${live.false_negative} FN over ${live.n} labelled requests`));
  } else {
    liveBox.append(el('div', 't', 'Live: no labelled traffic yet — poll the sandbox'));
  }
  liveBox.append(el('div', null, live.note || ''));
  b.append(liveBox);
  b.append(el('div', 'licence', bc.provenance || ''));

  // Semantic index
  const ix = $('#ml-index');
  panelTitle(ix, 'Semantic index');
  const si = m.semantic_index || {};
  const kv2 = el('dl', 'kv');
  const p2 = (k, v) => { kv2.append(el('dt', null, k)); kv2.append(el('dd', null, v)); };
  p2('documents', fmt(si.n_docs));
  p2('dimensions', si.dims || '—');
  p2('variance kept', si.explained_variance ?? '—');
  p2('rebuilt', si.age_s === null || si.age_s === undefined ? '—' : `${Math.round(si.age_s)}s ago`);
  ix.append(kv2);
  ix.append(el('div', 'licence', si.method || ''));

  // Tagger
  const tg = $('#ml-tagger');
  panelTitle(tg, 'Weak-supervision tagger');
  const t = m.tagger || {};
  const kv3 = el('dl', 'kv');
  const p3 = (k, v) => { kv3.append(el('dt', null, k)); kv3.append(el('dd', null, v)); };
  p3('rows', fmt(t.n_rows));
  p3('role coverage', t.role_coverage ?? '—');
  p3('role cv', t.role_family_cv ?? '—');
  p3('level coverage', t.seniority_coverage ?? '—');
  p3('level cv', t.seniority_cv ?? '—');
  tg.append(kv3);
  tg.append(el('div', 'licence', t.note || ''));

  // Populate the role filter from what the tagger actually produced.
  const sel = $('#f-role');
  if (sel.options.length <= 1) {
    ['ml-ai', 'data', 'backend', 'infra', 'frontend', 'mobile', 'security', 'product',
     'design', 'sales-ops'].forEach((r) => sel.append(new Option(r, r)));
  }
}

/* ---------------------------------------------------------------- theme */

function initTheme() {
  const stored = localStorage.getItem('sentinel-theme');
  if (stored) document.documentElement.dataset.theme = stored;
  $('#theme-toggle').onclick = () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('sentinel-theme', next);
  };
}

/* ----------------------------------------------------------- easter egg */

(() => {
  const SEQ = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft',
               'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
  let i = 0;
  addEventListener('keydown', (e) => {
    i = e.key === SEQ[i] || e.key.toLowerCase() === SEQ[i] ? i + 1 : 0;
    if (i === SEQ.length) {
      i = 0;
      document.body.classList.toggle('phosphor');
      const on = document.body.classList.contains('phosphor');
      pushEvent({
        ts: Date.now() / 1000, level: 'info', source: 'operator', kind: 'egg',
        message: on ? 'PHOSPHOR MODE ENGAGED — VT220 emulation active'
                    : 'phosphor mode disengaged',
      });
    }
  });
})();

/* ----------------------------------------------------------------- boot */

function boot() {
  initTheme();
  refreshHealth(); refreshSources(); refreshSandbox(); refreshJobs(); refreshML();
  startFeed();

  $('#q').addEventListener('input', debounceJobs);
  ['#f-source', '#f-role', '#f-seniority'].forEach((s) =>
    $(s).addEventListener('change', refreshJobs));

  $('#markup').onchange = async (e) => {
    await api('/sandbox/control', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ markup_version: Number(e.target.value) }),
    });
    refreshSandbox();
  };
  $('#poll-sandbox').onclick = async (e) => {
    e.target.disabled = true;
    try { await api('/api/sources/sandbox/poll', { method: 'POST' }); }
    finally { e.target.disabled = false; refreshSources(); refreshSandbox(); refreshJobs(); }
  };
  $('#reset-sandbox').onclick = async () => {
    await api('/sandbox/control', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ reset: true }),
    });
    await api('/api/sources/sandbox/reset', { method: 'POST' });
    refreshSandbox(); refreshSources();
  };

  // Staggered so a cold container is not hit by five concurrent requests the
  // instant it wakes up.
  setInterval(refreshHealth, 5000);
  setInterval(refreshSources, 6000);
  setInterval(refreshSandbox, 7000);
  setInterval(refreshJobs, 15000);
  setInterval(refreshML, 30000);
}

document.readyState === 'loading'
  ? addEventListener('DOMContentLoaded', boot)
  : boot();
