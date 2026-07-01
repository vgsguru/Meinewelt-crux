#!/usr/bin/env python3
"""
Redrob India Runs — Intelligent Candidate Discovery & Ranking
Offline, CPU-only, no-network ranker for the "Senior AI Engineer — Founding Team" JD.

Hybrid design (matches the webinar + submission_spec constraints):
  * NO hosted-LLM calls. Classical IR + local sentence-embeddings + rule-based reasoning.
    Runs on CPU in minutes for 100k candidates, <16GB RAM, no GPU, no network.
  * "Reads between the lines" in TWO complementary ways:
      - SEMANTIC: cosine similarity of a local MiniLM embedding of the candidate's career
        narrative against the JD embedding (pre-computed by precompute_embeddings.py). This
        catches "built a predictive orchestrator for supply chain" ~ ranking/ML with no keywords.
      - LEXICAL: weighted BM25-style match of JD capability concepts over the same narrative.
  * Applies the JD's explicit traps & disqualifiers (wrong-title keyword stuffers,
    services-only careers, CV/speech-without-NLP, pure-research-without-production).
  * Weighs behavioral signals as an availability modifier (down-weight perfect-on-paper
    candidates who are inactive / never respond).
  * Detects honeypots (impossible profiles) and forces them out of the top.

If the pre-computed embeddings are absent, it degrades gracefully to lexical-only (still valid).

Usage:
  python precompute_embeddings.py --candidates ./candidates.jsonl   # one-time (pre-compute)
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""
import argparse, csv, json, os, re, sys
from datetime import date

TODAY = date(2026, 6, 1)  # pinned for deterministic, network-free recency scoring

CONCEPTS = {
    "retrieval":      (3.0, ["retrieval", "information retrieval", " ir ", "semantic search", "search relevance", "search ranking", "elasticsearch", "opensearch", "solr", "lucene", "bm25"]),
    "embeddings":     (3.0, ["embedding", "embeddings", "sentence-transformer", "sentence transformer", " bge", " e5 ", "word2vec", "vectoriz"]),
    "vector_db":      (2.6, ["vector database", "vector search", "vector store", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "annoy", "hnsw", "nearest neighbor", "approximate nearest"]),
    "ranking_recsys": (3.0, ["ranking", "re-rank", "rerank", "learning to rank", "learning-to-rank", "ltr", "recommendation", "recommender", "recsys", "personaliz", "relevance rank", "candidate ranking"]),
    "nlp":            (2.4, ["nlp", "natural language", "text classification", "named entity", "ner", "question answering", "information extraction", "text mining", "semantic"]),
    "llm":            (1.8, ["large language model", "llm", "language model", "transformer", "bert", "gpt", "rag", "retrieval augmented", "fine-tun", "finetun", "lora", "qlora", "peft"]),
    "evaluation":     (2.4, ["ndcg", " mrr", " map@", "mean average precision", "precision@", "recall@", "a/b test", "ab test", "offline metric", "evaluation framework", "eval framework", "relevance judgment"]),
    "python":         (1.2, ["python"]),
    "production":     (1.8, ["production", "deployed to", "real users", "in production", "serving", "low latency", "at scale", "throughput", "productioniz"]),
    "ml_general":     (1.0, ["machine learning", " ml ", "deep learning", "pytorch", "tensorflow", "scikit", "xgboost", "gradient boost", "model training", "feature engineering"]),
    "data_eng":       (0.7, ["data pipeline", "etl", "spark", "kafka", "airflow", "streaming"]),
}

GOOD_TITLE = re.compile(r"\b(ml|machine learning|ai|a\.i\.|artificial intelligence|nlp|data scien|applied scien|research (engineer|scien)|search|relevance|recommendation|ranking|software engineer|backend|full[- ]?stack|platform engineer|deep learning|computer scien|staff engineer|principal engineer)\b", re.I)
BAD_TITLE = re.compile(r"\b(marketing|sales|business development|bd manager|account manager|recruit|talent acquisition|hr |human resource|content writer|copywriter|editor|teacher|professor|lecturer|nurse|doctor|accountant|finance manager|operations manager|customer support|customer success|mechanical|civil|electrical engineer|graphic design|ui/ux design|product manager|project manager|program manager|scrum master|qa tester|manual test|network admin|system admin|devops|sre|site reliab)\b", re.I)

SERVICES_COMPANIES = re.compile(r"\b(tcs|tata consultancy|infosys|wipro|accenture|cognizant|capgemini|hcl|tech mahindra|mindtree|mphasis|ltimindtree|l&t infotech|igate|syntel|hexaware|birlasoft|persistent systems|nttdata|ntt data|dxc|virtusa)\b", re.I)
CVSR = re.compile(r"\b(computer vision|image classif|object detection|opencv|speech recognition|asr|text-to-speech|tts|robotics|slam|lidar|point cloud)\b", re.I)
RESEARCH_ONLY = re.compile(r"\b(phd|ph\.d|postdoc|post-doc|research fellow|research assistant|research scholar|academic)\b", re.I)
PRODUCT_HINT = re.compile(r"\b(users?|customers?|product|scale|revenue|latency|deployed|shipped|a/b|traffic|dau|mau)\b", re.I)
LANGCHAIN_ONLY = re.compile(r"\blangchain\b", re.I)
GOOD_LOCATIONS = re.compile(r"\b(pune|noida|bengaluru|bangalore|hyderabad|mumbai|delhi|gurgaon|gurugram|ncr|india)\b", re.I)


def f(x, d=0.0):
    try: return float(x)
    except: return d

def i(x, d=0):
    try: return int(float(x))
    except: return d

def as_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")

def parse_date(s):
    try:
        y, m, d = str(s)[:10].split("-"); return date(int(y), int(m), int(d))
    except: return None


def candidate_text(c):
    p = c.get("profile", {}) or {}
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", ""), p.get("current_industry", "")]
    for role in (c.get("career_history") or []):
        parts += [role.get("title", ""), role.get("description", ""), role.get("industry", "")]
    for sk in (c.get("skills") or []):
        parts.append(sk.get("name", ""))
    txt = " ".join(str(x) for x in parts if x).lower()
    return " " + re.sub(r"\s+", " ", txt) + " "


def concept_scores(text, doclen):
    k = 2.0
    out = {}
    for name, (w, forms) in CONCEPTS.items():
        tf = sum(text.count(form) for form in forms)
        if tf:
            norm = tf * (k + 1) / (tf + k * (0.5 + 0.5 * doclen / 900.0))
            out[name] = w * norm
    return out


def experience_fit(yoe):
    if 5 <= yoe <= 9:   return 1.0
    if 4 <= yoe < 5:    return 0.85
    if 9 < yoe <= 11:   return 0.85
    if 3 <= yoe < 4:    return 0.6
    if 11 < yoe <= 14:  return 0.7
    if yoe < 3:         return 0.3
    return 0.6


def signal_modifier(sig):
    if not sig: return 0.85
    resp = f(sig.get("recruiter_response_rate"), 0.3)
    icr = f(sig.get("interview_completion_rate"), 0.5)
    oar = f(sig.get("offer_acceptance_rate"), 0.5)
    compl = f(sig.get("profile_completeness_score"), 60) / 100.0
    otw = as_bool(sig.get("open_to_work_flag"))
    la = parse_date(sig.get("last_active_date"))
    recency = 1.0
    if la:
        days = (TODAY - la).days
        recency = 1.0 if days <= 30 else 0.9 if days <= 90 else 0.75 if days <= 180 else 0.55
    base = 0.55 + 0.22 * resp + 0.13 * icr + 0.05 * oar + 0.05 * compl
    base *= recency
    if otw: base += 0.06
    return max(0.5, min(1.15, base))


def is_honeypot(c):
    p = c.get("profile", {}) or {}
    yoe = f(p.get("years_of_experience"))
    hist = c.get("career_history") or []
    for sk in (c.get("skills") or []):
        if str(sk.get("proficiency", "")).lower() in ("expert", "advanced") and i(sk.get("duration_months")) == 0:
            return True
    total_months = sum(i(r.get("duration_months")) for r in hist)
    if yoe > 0 and total_months > (yoe + 2) * 12 + 6:
        return True
    for r in hist:
        if i(r.get("duration_months")) > (yoe * 12 + 30):
            return True
    return False


def components(c):
    """Return the score building blocks for one candidate (pre-normalization)."""
    p = c.get("profile", {}) or {}
    sig = c.get("redrob_signals") or {}
    text = candidate_text(c)
    doclen = len(text.split())
    cs = concept_scores(text, doclen)

    core = (cs.get("retrieval", 0) + cs.get("embeddings", 0) + cs.get("vector_db", 0)
            + cs.get("ranking_recsys", 0) + cs.get("nlp", 0) + cs.get("evaluation", 0))
    support = (cs.get("llm", 0) + cs.get("production", 0) + cs.get("python", 0)
               + cs.get("ml_general", 0) + cs.get("data_eng", 0))
    lex_raw = core + 0.5 * support

    yoe = f(p.get("years_of_experience"))
    exp = experience_fit(yoe)

    title_good = bool(GOOD_TITLE.search(" ".join([p.get("current_title", "")] + [r.get("title", "") for r in (c.get("career_history") or [])])))
    ct = p.get("current_title", "") or ""
    title_bad = bool(BAD_TITLE.search(ct)) and not GOOD_TITLE.search(ct)

    hist = c.get("career_history") or []
    services_ratio = (sum(1 for r in hist if SERVICES_COMPANIES.search(r.get("company", "") or "")) / len(hist)) if hist else 0.0
    has_nlp_ir = bool(cs.get("nlp") or cs.get("retrieval") or cs.get("embeddings") or cs.get("ranking_recsys"))
    cvsr = bool(CVSR.search(text)) and not has_nlp_ir
    research = bool(RESEARCH_ONLY.search(text)) and not (bool(PRODUCT_HINT.search(text)) or cs.get("production", 0) > 0)

    mult = 1.0
    if title_good: mult *= 1.12
    if title_bad:  mult *= 0.35
    if services_ratio >= 0.8: mult *= 0.6
    elif services_ratio >= 0.5: mult *= 0.85
    if cvsr: mult *= 0.55
    if research: mult *= 0.55
    if LANGCHAIN_ONLY.search(text) and not (cs.get("retrieval") or cs.get("ranking_recsys") or cs.get("evaluation")):
        mult *= 0.8
    if GOOD_LOCATIONS.search((p.get("location", "") or "") + " " + (p.get("country", "") or "")) or as_bool(sig.get("willing_to_relocate")):
        mult *= 1.05

    sig_mod = signal_modifier(sig)
    honeypot = is_honeypot(c)

    meta = {"yoe": yoe, "cs": cs, "title_good": title_good, "title_bad": title_bad,
            "services_ratio": services_ratio, "cvsr": cvsr, "research": research,
            "sig": sig, "title": ct}
    return lex_raw, exp, mult, sig_mod, honeypot, meta


def minmax(vals):
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return [(v - lo) / rng for v in vals]


TOPLABEL = {"retrieval": "retrieval/search", "embeddings": "embeddings", "vector_db": "vector search",
            "ranking_recsys": "ranking/recsys", "nlp": "NLP", "evaluation": "ranking evaluation",
            "llm": "LLMs", "production": "production ML", "ml_general": "applied ML"}

def reasoning(meta, strong_semantic):
    yoe = meta["yoe"]; cs = meta["cs"]
    strengths = sorted([(v, k) for k, v in cs.items() if k in TOPLABEL], reverse=True)[:3]
    strs = [TOPLABEL[k] for _, k in strengths]
    title = meta["title"] or "engineer"
    sig = meta["sig"]
    resp = f(sig.get("recruiter_response_rate"), -1)

    if strs:
        lead = f"{title} with {yoe:.1f} yrs; career history shows {', '.join(strs)}"
    elif strong_semantic:
        lead = f"{title} with {yoe:.1f} yrs; project narrative reads as strong applied-ML/ranking work"
    else:
        lead = f"{title} with {yoe:.1f} yrs of adjacent engineering experience"
    if meta["title_good"] and strs:
        lead += " — fits the retrieval/ranking mandate"

    concerns = []
    if meta["services_ratio"] >= 0.5: concerns.append("services-heavy background (JD prefers product companies)")
    if meta["cvsr"]: concerns.append("vision/speech-leaning, lighter NLP/IR")
    if meta["research"]: concerns.append("research-leaning, limited production signal")
    if 0 <= resp < 0.25: concerns.append(f"low recruiter response ({resp:.2f})")
    la = sig.get("last_active_date")
    if la and parse_date(la) and (TODAY - parse_date(la)).days > 120:
        concerns.append("inactive recently")

    r = lead + "."
    if concerns:
        r += " Concern: " + "; ".join(concerns[:2]) + "."
    elif resp >= 0.5:
        r += f" Responsive (recruiter response {resp:.2f})."
    return r[:300]


def load_embeddings(here, ids_order):
    """Load pre-computed embeddings; return dict id->semantic_sim in [0,1] or None."""
    try:
        import numpy as np
        emb = np.load(os.path.join(here, "cand_emb.npy")).astype(np.float32)
        jd = np.load(os.path.join(here, "jd_vec.npy")).astype(np.float32)
        with open(os.path.join(here, "cand_ids.json"), "r", encoding="utf-8") as fh:
            emb_ids = json.load(fh)
        sims = emb @ jd  # both L2-normalized -> cosine in [-1, 1]
        sims = (sims + 1.0) / 2.0
        return {cid: float(s) for cid, s in zip(emb_ids, sims)}
    except Exception as e:
        print(f"[rank] embeddings not used ({type(e).__name__}); lexical-only fallback.", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--w-semantic", type=float, default=0.55)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    recs = []
    with open(args.candidates, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: c = json.loads(line)
            except: continue
            if not c.get("candidate_id") or not c.get("profile"): continue
            recs.append(c)

    ids = [c["candidate_id"] for c in recs]
    comp = [components(c) for c in recs]
    lex_raw = [x[0] for x in comp]

    sem_map = load_embeddings(here, ids)
    have_sem = sem_map is not None
    sem_raw = [sem_map.get(cid, 0.0) for cid in ids] if have_sem else [0.0] * len(ids)

    lex_n = minmax(lex_raw)
    sem_n = minmax(sem_raw) if have_sem else [0.0] * len(ids)
    ws = args.w_semantic if have_sem else 0.0
    wl = 1.0 - ws

    scored = []
    for idx, c in enumerate(recs):
        _, exp, mult, sig_mod, honeypot, meta = comp[idx]
        base = wl * lex_n[idx] + ws * sem_n[idx]
        score = 0.0 if honeypot else base * (0.55 + 0.45 * exp) * mult * sig_mod
        strong_sem = have_sem and sem_n[idx] > 0.7
        scored.append((score, ids[idx], meta, strong_sem))

    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:args.topn]

    mx = top[0][0] or 1.0
    rows, prev = [], 2.0
    for rank, (s, cid, meta, strong_sem) in enumerate(top, 1):
        val = round(0.15 + 0.84 * (s / mx), 4) if mx > 0 else 0.15
        if val >= prev:
            val = round(prev - 0.0001, 4)
        prev = val
        rows.append([cid, rank, f"{val:.4f}", reasoning(meta, strong_sem)])

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        w.writerows(rows)

    print(f"Scored {len(recs):,} candidates ({'hybrid: embeddings + lexical + rules' if have_sem else 'lexical + rules (no embeddings file)'})")
    print(f"Wrote top {len(rows)} to {args.out} | honeypots zeroed: {sum(1 for s,_,_,_ in scored if s==0.0):,}")
    for cid, rank, val, why in rows[:5]:
        print(f"  {rank:>3} {cid} {val}  {why}")


if __name__ == "__main__":
    main()
