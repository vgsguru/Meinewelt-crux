#!/usr/bin/env python3
"""
Crux — India Runs offline ranker: hosted SANDBOX (Streamlit), styled to match the Crux web app.

A judge/organizer can upload a small candidate sample (.json array or .jsonl), edit the role,
and get the same offline hybrid ranking the submission uses — then download the CSV. Runs entirely
on CPU; the only model is a small LOCAL sentence-transformer (no hosted LLM).

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud (repo main file: app.py)
"""
import io, os, json, csv, base64
import streamlit as st

import rank as R
from precompute_embeddings import narrative, JD_QUERY

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Crux · Offline Candidate Ranker", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------- brand theme
CRUX_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
:root { --fg:#1C1C1C; --muted:#6E6E6E; --border:rgba(28,28,28,.10); --bg:#FCFCFC; }
html, body, [class*="css"], .stApp { background:var(--bg); color:var(--fg); font-family:'Inter',ui-sans-serif,system-ui,sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility:hidden; }
h1,h2,h3,h4 { font-family:'Space Grotesk',ui-sans-serif,sans-serif !important; letter-spacing:-0.02em; font-weight:700; color:var(--fg); }
.crux-badge { display:inline-flex; align-items:center; gap:.5rem; background:rgba(28,28,28,.06); color:var(--fg);
  padding:.28rem .8rem; border-radius:999px; font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.12em; }
.crux-card { background:#fff; border:1px solid var(--border); border-radius:1.25rem; padding:1.15rem 1.3rem;
  box-shadow:0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.04); }
.crux-sub { color:var(--muted); font-size:.9rem; }
/* pill buttons + inputs, Crux-style */
.stButton>button, .stDownloadButton>button { background:var(--fg); color:#fff; border:none; border-radius:999px;
  padding:.6rem 1.4rem; font-weight:600; font-family:'Space Grotesk',sans-serif; transition:transform .12s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform:scale(1.03); color:#fff; background:#000; }
.stTextArea textarea, .stNumberInput input { border-radius:.9rem !important; border:1px solid var(--border) !important; }
[data-testid="stFileUploaderDropzone"] { border-radius:1.1rem; border:1px dashed var(--border); background:#fff; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:1rem; overflow:hidden; }
.block-container { padding-top:2.2rem; max-width:1180px; }
hr { border-color:var(--border); }
</style>
"""
st.markdown(CRUX_CSS, unsafe_allow_html=True)


def logo_b64():
    try:
        with open(os.path.join(HERE, "assets", "logo.png"), "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def get_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    m.max_seq_length = 128
    return m


def read_candidates(raw_bytes):
    text = raw_bytes.decode("utf-8", "ignore")
    recs = []
    if text.lstrip().startswith("["):
        recs = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                try: recs.append(json.loads(line))
                except Exception: pass
    return [c for c in recs if c and c.get("candidate_id") and c.get("profile")]


def rank_live(recs, query, use_embeddings, w_sem=0.55):
    ids = [c["candidate_id"] for c in recs]
    comp = [R.components(c) for c in recs]
    lex_raw = [x[0] for x in comp]
    if use_embeddings and len(recs) > 1:
        model = get_model()
        cand_vecs = model.encode([narrative(c) for c in recs], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        qv = model.encode([query or JD_QUERY], normalize_embeddings=True, convert_to_numpy=True)[0]
        sem_n = R.minmax(list(((cand_vecs @ qv) + 1.0) / 2.0))
        ws, wl = w_sem, 1.0 - w_sem
    else:
        sem_n = [0.0] * len(recs); ws, wl = 0.0, 1.0
    lex_n = R.minmax(lex_raw)
    scored = []
    for idx in range(len(recs)):
        _, exp, mult, sig_mod, honeypot, meta = comp[idx]
        base = wl * lex_n[idx] + ws * sem_n[idx]
        score = 0.0 if honeypot else base * (0.55 + 0.45 * exp) * mult * sig_mod
        scored.append((score, ids[idx], meta, ws > 0 and sem_n[idx] > 0.7))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored


# --------------------------------------------------------------------------- header
lb = logo_b64()
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.2rem;">
      {'<img src="data:image/png;base64,'+lb+'" style="height:40px;width:auto;"/>' if lb else ''}
      <span style="font-family:'Space Grotesk';font-weight:700;font-size:1.9rem;letter-spacing:-.02em;">Crux</span>
      <span class="crux-badge" style="margin-left:.4rem;">Offline Ranker</span>
    </div>
    <p class="crux-sub" style="margin:.1rem 0 1.1rem;">India Runs · Intelligent Candidate Discovery &amp; Ranking —
    the constraint-compliant submission engine. CPU-only, <b>no hosted LLM</b>: local embeddings + classical IR + rules.</p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    if lb:
        st.markdown(f'<img src="data:image/png;base64,{lb}" style="height:34px;margin-bottom:.6rem;"/>', unsafe_allow_html=True)
    st.markdown("#### Role / Job Description")
    query = st.text_area("What the role genuinely needs", value=JD_QUERY, height=230, label_visibility="collapsed")
    use_emb = st.toggle("Semantic embeddings (local MiniLM)", value=True,
                        help="Blend a local sentence-embedding similarity with the lexical+rules score.")
    w_sem = st.slider("Semantic weight", 0.0, 1.0, 0.55, 0.05, disabled=not use_emb)
    topn = st.number_input("Top N", 1, 100, 20)

up = st.file_uploader("Upload candidate sample (.json array or .jsonl)", type=["json", "jsonl"])
st.markdown('<p class="crux-sub">Use <code>sample_candidates.json</code> from the challenge bundle, or any slice '
            'of <code>candidates.jsonl</code>. The full 100k run uses <code>rank.py</code> with pre-computed embeddings.</p>',
            unsafe_allow_html=True)

if up is not None:
    try:
        recs = read_candidates(up.getvalue())
    except Exception as e:
        st.error(f"Couldn't parse that file: {e}"); recs = []
    if recs:
        st.success(f"Loaded {len(recs):,} candidates from {up.name}")
        if st.button("Rank candidates"):
            with st.spinner("Ranking — embeddings + lexical + rules…"):
                scored = rank_live(recs, query, use_emb, w_sem)
            top = scored[: int(topn)]
            mx = top[0][0] or 1.0
            table, csv_rows, prev = [], [], 2.0
            for rk, (s, cid, meta, strong) in enumerate(top, 1):
                val = round(0.15 + 0.84 * (s / mx), 4) if mx > 0 else 0.15
                if val >= prev: val = round(prev - 0.0001, 4)
                prev = val
                why = R.reasoning(meta, strong)
                table.append({"rank": rk, "candidate_id": cid, "title": meta["title"],
                              "yoe": round(meta["yoe"], 1), "score": val, "reasoning": why})
                csv_rows.append([cid, rk, f"{val:.4f}", why])
            st.markdown(f"### Top {len(table)}")
            st.dataframe(table, use_container_width=True, hide_index=True)
            buf = io.StringIO(); w = csv.writer(buf)
            w.writerow(["candidate_id", "rank", "score", "reasoning"]); w.writerows(csv_rows)
            st.download_button("⬇  Download ranking CSV", buf.getvalue(), file_name="submission.csv", mime="text/csv")
            zeroed = sum(1 for s, _, _, _ in scored if s == 0.0)
            if zeroed:
                st.info(f"{zeroed} candidate(s) detected as honeypots / impossible profiles and forced out of contention.")
else:
    st.markdown('<div class="crux-card crux-sub">Upload a file to begin. Nothing is sent anywhere — ranking runs locally on CPU.</div>',
                unsafe_allow_html=True)
