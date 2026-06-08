# Foundational SAT Skill Gap Prediction Using Kernel k-NN and Label Propagation

**Standalone prototype — synthetic data only.** Not connected to Reading Rooms production.

Predicts which SAT Reading/Writing skills a student is likely weak in when those skills have **not been tested yet**. Skill scores are **computed bottom-up from synthetic MCQ item responses** — every percentage is auditable from visible question attempts.

## GitHub

**Repository:** [github.com/maxlitvinenka/sat-skill-gap](https://github.com/maxlitvinenka/sat-skill-gap)

### Try it in the browser (no install)

**Live demo:** [maxlitvinenka.github.io/sat-skill-gap](https://maxlitvinenka.github.io/sat-skill-gap/)

Opens the full dashboard with student selector, methodology walkthrough, recommendations, and charts. Data is pre-generated — no `git clone` or localhost required.

> The **Live showcase** panel (custom students, σ/α/k sliders, Python API) only works when you run locally with `npm run dev:live`. GitHub Pages hosts the static dashboard only.

### Run locally (optional)

```bash
git clone https://github.com/maxlitvinenka/sat-skill-gap.git
cd sat-skill-gap
```

## Quick start

### Python pipeline

```bash
cd sat-skill-gap
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py --export-frontend
```

Generates:
- `data/item_bank.csv` — 216 synthetic MCQs (3 per skill)
- `data/student_responses.csv` — every item attempt (stem, choice, correct/incorrect)
- `data/synthetic_student_scores.csv` — skill mastery (% correct), only for tested skills
- `output/predicted_missing_skills.csv` — neighbor, related, and final predictions for all untested skills
- `output/recommendations_*.csv` — ranked gap predictions
- `output/figures/*.png` — heatmap, kernel neighbor heatmap, bar charts
- `frontend/public/data/dashboard.json` — dashboard data with `observedWork`

### Localhost dashboard (static)

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — reads pre-generated `dashboard.json`.

### Live showcase (Python API + dashboard)

Run the FastAPI server and Vite together so predictions execute on demand via real `lib.py`:

```bash
cd sat-skill-gap
source .venv/bin/activate
pip install -r requirements.txt
npm install
npm run dev:live
```

Open **http://localhost:5173** — a **Live mode** badge appears when the API is connected.

The **Live showcase** panel lets you:
- Change random seed, σ, α, and k — then **Run analysis** to refresh recommendations
- **Regenerate cohort** with a new seed
- **Add a custom student** (name + tested skills + mastery %) and analyze against the cohort
- View a **computation trace** showing which `lib.py` functions ran and the numeric results

API endpoints: `GET /api/health`, `GET /api/skills`, `POST /api/pipeline`, `POST /api/analyze`.

### For school project reviewers

The dashboard includes **"How this data is built"** with KaTeX formulas and live worked examples per student. In live mode, the same Python pipeline powers the UI — not pre-baked JSON alone.

## Data flow (response-based)

```
Item bank → Student MCQ attempts → Response matrix R
R aggregated via Q → Skill matrix S (% correct per skill)
Untested skills = no items attempted
Gaussian kernel k-NN on S → label propagation via skill graph W → priority ranking
```

## Linear algebra

| Concept | Where | How |
|---------|-------|-----|
| Response matrix R | `lib.build_response_matrix()` | students × items, 0/1/NaN |
| Skill-item matrix Q | `lib.build_skill_item_matrix()` | items × skills, one-hot |
| Skill matrix S | `responses_to_skill_matrix()` | % correct per (student, skill) |
| Squared distance | `kernel_similarity_observed()` | `(u−v)ᵀ(u−v)` on shared skills Ω |
| Gaussian kernel | `gaussian_kernel()` | `exp(−‖u−v‖² / 2σ²)` |
| Skill affinity W | `build_skill_affinity_matrix()` | prerequisites, category, level neighbors |
| k-NN prediction | `predict_missing_scores_knn_propagate()` | kernel-weighted peer average |
| Label propagation | `related_skill_prediction()` | weighted average over related known skills |
| Blend | same | `α·neighbor + (1−α)·related` |

Latent ability profiles are used **only** to set P(correct) when generating synthetic responses — they are not the displayed score.

## Algorithm

1. **100 students**, each tested on **~40 skills** (38–42), all 3 items per tested skill
2. Simulate MCQ responses from latent ability + item difficulty
3. **S[i,j] = 100 × (# correct / # attempted)** for tested skills
4. Build skill affinity graph **W** (prerequisites, same category, adjacent level)
5. **Gaussian kernel** on shared known skills; **k=15** nearest neighbors
6. For each missing skill: blend neighbor prediction (α=0.7) with related-skill propagation
7. Priority: `(100 - predicted) × foundational_weight` for untested skills

## Assignment requirements

1. **Non-trivial linear algebra:** R, Q, S matrices; student vectors; distances/norms; Gaussian kernel; weighted k-NN; label propagation; missing-entry prediction.
2. **Real problem:** Cannot test every SAT skill monthly — surface hidden foundational gaps.
3. **Verifiable artifacts:** Item bank, response log, derived scores, `predicted_missing_skills.csv`, notebook, PNGs, dashboard JSON.

Optional future extension: matrix factorization (PCA/SVD) — not used in this prototype.

## Project structure

```
items.py               # 216-item MCQ bank
skills.py              # 72-skill catalog
lib.py                 # response generation + kernel k-NN + propagation
predict.py             # CLI
export_frontend.py     # dashboard JSON with observedWork
sat_skill_gap_prediction.ipynb
frontend/              # Reading Rooms styled demo UI
data/                  # item_bank, student_responses, skill scores
output/                # recommendations + predicted_missing_skills + figures
```
