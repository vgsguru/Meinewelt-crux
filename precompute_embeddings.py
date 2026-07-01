#!/usr/bin/env python3
"""
PRE-COMPUTATION step (allowed to exceed the 5-min ranking budget; may use network once
to fetch the local model, then runs offline). Encodes every candidate's career narrative
into a 384-dim sentence embedding with a small LOCAL model (all-MiniLM-L6-v2), plus the
JD embedding. rank.py then consumes these as pre-computed artifacts and does a pure-numpy
cosine similarity at ranking time — no torch, no network, milliseconds for 100k.

Outputs (next to this script):
  cand_emb.npy    float16 [N, 384], L2-normalized
  cand_ids.json   [N] candidate_ids aligned to cand_emb rows
  jd_vec.npy      float16 [384], L2-normalized JD embedding

Usage:
  python precompute_embeddings.py --candidates ./candidates.jsonl
"""
import argparse, json, os, re, sys, time
import numpy as np

MODEL = "all-MiniLM-L6-v2"

# A distilled statement of what the JD *means* — the ideal candidate — used as the query
# vector. Written as capability prose so the embedding sits near real practitioner language.
JD_QUERY = (
    "Senior AI engineer, 5 to 9 years, who has shipped end-to-end ranking, retrieval, search "
    "and recommendation systems to real users in production at a product company. Deep, hands-on "
    "experience with embeddings-based retrieval (sentence-transformers, BGE, E5), vector databases "
    "and hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch), learning-to-rank and "
    "re-ranking, NLP and information retrieval, and rigorous ranking evaluation with NDCG, MRR and "
    "A/B testing. Strong Python and production ML systems. Scrappy product engineer, not a pure "
    "researcher, not a consulting-services generalist."
)


def narrative(c):
    # Front-load the signal that reveals *what they actually built* (title + career
    # descriptions) so it survives the model's sequence-length truncation, then the
    # summary and skills. This ordering is what lets embeddings "read between the lines".
    p = c.get("profile", {}) or {}
    parts = [p.get("current_title", ""), p.get("headline", "")]
    for r in (c.get("career_history") or [])[:4]:
        parts += [r.get("title", ""), r.get("description", "")]
    parts.append((p.get("summary", "") or "")[:400])
    parts.append(", ".join(s.get("name", "") for s in (c.get("skills") or [])[:15]))
    txt = " ".join(str(x) for x in parts if x)
    return re.sub(r"\s+", " ", txt).strip()[:2000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    from sentence_transformers import SentenceTransformer
    t = time.time()
    model = SentenceTransformer(MODEL)
    model.max_seq_length = 64  # front-loaded narrative fits here; ~2x faster than 128 on CPU
    print(f"[precompute] loaded {MODEL} in {time.time()-t:.1f}s")

    ids, texts = [], []
    with open(args.candidates, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            if not c.get("candidate_id") or not c.get("profile"):
                continue
            ids.append(c["candidate_id"])
            texts.append(narrative(c))
    print(f"[precompute] encoding {len(texts):,} candidate narratives ...")

    t = time.time()
    emb = model.encode(texts, batch_size=args.batch, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    print(f"[precompute] encoded in {(time.time()-t)/60:.1f} min")

    jd = model.encode([JD_QUERY], normalize_embeddings=True, convert_to_numpy=True)[0]

    np.save(os.path.join(here, "cand_emb.npy"), emb.astype(np.float16))
    np.save(os.path.join(here, "jd_vec.npy"), jd.astype(np.float16))
    with open(os.path.join(here, "cand_ids.json"), "w", encoding="utf-8") as fh:
        json.dump(ids, fh)
    print(f"[precompute] wrote cand_emb.npy {emb.shape}, jd_vec.npy, cand_ids.json")


if __name__ == "__main__":
    main()
