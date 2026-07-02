# Meinewelt-Crux — Data & Al Challenge : Intelligent Candidate Discovery

**Test here**: https://meinewelt-crux.streamlit.app/

A constraint-compliant, **offline** ranker for the **Senior AI Engineer — Founding Team** job
description. It ranks the top 100 candidates from `candidates.jsonl` (100k pool) with grounded
reasoning, using a **hybrid of local sentence-embeddings + classical IR + rules** — **no hosted
LLM**, CPU-only, no network at ranking time.

## Reproduce the submission

```bash
# 1. Pre-computation (allowed to exceed the 5-min budget; fetches a small local model once).
python precompute_embeddings.py --candidates ./candidates.jsonl
#    -> cand_emb.npy, cand_ids.json, jd_vec.npy   (~6 min for 100k on CPU)

# 2. Ranking step (offline, pure numpy over the pre-computed vectors + rules).
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
#    -> ~90s for 100k on CPU

# 3. Validate against the official spec validator.
python validate_submission.py submission.csv        # -> "Submission is valid."
```

`rank.py` **degrades gracefully to lexical-only** (Python standard library, no numpy needed) if
the embedding files are absent — so the ranking step always produces a valid CSV.

## Try it live (sandbox)

```bash
python app.py     # self-bootstraps the UI and opens your browser — no streamlit command needed
```

On Windows you can simply **double-click `run_app.bat`** — it installs dependencies on first run
and starts the app. (`streamlit run app.py` still works too.)

Upload a candidate sample (`sample_candidates.json` or any slice of `candidates.jsonl`), edit the
role, and get the same ranking with a downloadable CSV. Deployable free on Streamlit Community Cloud.

> Note: the UI is optional — the actual ranking engine is the `rank.py` CLI, which has no UI
> dependencies at all. First UI run downloads the small local MiniLM model (~90 MB) once.

## Why this design (per the challenge rules)

- **No hosted-LLM calls during ranking.** The spec hard-disqualifies any OpenAI/Anthropic/Gemini/
  Groq call at Stage 3; the ranking step is reproduced in a sandboxed 5-min / 16GB / CPU / no-network
  container. This ranker uses only a small **local** MiniLM embedding (pre-computed) + numpy + rules.
- **Reads between the lines — two ways.**
  - **Semantic:** cosine similarity of a local `all-MiniLM-L6-v2` embedding of the candidate's
    **career narrative** against the JD embedding. Catches "built an end-to-end predictive
    orchestrator for supply chain" ≈ ranking/ML with zero buzzwords.
  - **Lexical:** weighted, BM25-style match of JD capability concepts (retrieval, embeddings, vector
    search, ranking/recsys, NLP, ranking-evaluation) over the same narrative.
- **JD traps & disqualifiers applied.** An AI-skill-stuffed "Marketing Manager" is penalized (title
  gate, 0.35×); services-only careers, CV/speech-without-NLP, research-without-production, and
  LangChain-wrapper-only "AI experience" are down-weighted — straight from the JD's "we do NOT want".
- **Behavioral signals as an availability modifier.** Recruiter response rate, interview completion,
  recency, open-to-work, completeness — the JD's "perfect on paper but never responds → not hireable".
- **Honeypot defense.** Internally-contradictory profiles (expert skill with 0 months used, tenure
  exceeding total career) are detected and zeroed so they can't reach the top-100.

## Scoring

```
base       = 0.45·lexical_norm + 0.55·semantic_norm          # both min-max normalized across the pool
final      = base · (0.55 + 0.45·experience_band_fit) · title_gate · career_penalties · signal_modifier
honeypots  -> 0
```
Output scores are normalized strictly non-increasing by rank; ties broken by `candidate_id` ascending (spec §3).

## Files

| File | Purpose |
|---|---|
| `rank.py` | The ranking step. Produces `submission.csv`. |
| `precompute_embeddings.py` | One-time embedding pre-computation (the "pre-computed artifact" the spec allows). |
| `app.py` | Streamlit sandbox — upload a sample and rank it live. |
| `validate_submission.py` | Official challenge validator (bundled for convenience). |
| `submission_metadata.yaml` | Portal metadata mirror. |

## Relationship to the Crux web app(For Ideathon-Track-3)
The streamlit live version (https://meinewelt-crux.streamlit.app/)
The live product (https://crux-beta.vercel.app → Recruiter → **Talent Discovery**) is the interactive
demo. It uses an LLM re-rank for interactive polish, which is fine for a demo but is **not** the
competition path — the offline `rank.py` here is the constraint-compliant engine, and `app.py` is its
sandbox.
