# Foundational SAT Skill Gap Prediction

**Standalone prototype — synthetic data only.** Not connected to Reading Rooms production.

Predicts which SAT Reading/Writing skills a student is likely weak in when those skills have **not been tested yet**. Skill scores are **computed bottom-up from synthetic MCQ item responses** — every percentage is auditable from visible question attempts.

## Quick start

### Python pipeline

```bash
cd sat-skill-gap-prototype
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py --export-frontend
```

Generates:
- `data/item_bank.csv` — 216 synthetic MCQs (3 per skill)
- `data/student_responses.csv` — every item attempt (stem, choice, correct/incorrect)
- `data/synthetic_student_scores.csv` — skill mastery (% correct), only for tested skills
- `output/recommendations_*.csv` — ranked gap predictions
- `output/figures/*.png` — heatmap and bar charts
- `frontend/public/data/dashboard.json` — dashboard data with `observedWork`

### Localhost dashboard

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — shows tested skills, expandable item responses, then predictions.

## Data flow (response-based)

```
Item bank → Student MCQ attempts → Response matrix R
R aggregated via Q → Skill matrix S (% correct per skill)
Untested skills = no items attempted
Cosine similarity on S → predict missing skills → priority ranking
```

## Linear algebra

| Concept | Where | How |
|---------|-------|-----|
| Response matrix R | `lib.build_response_matrix()` | students × items, 0/1/NaN |
| Skill-item matrix Q | `lib.build_skill_item_matrix()` | items × skills, one-hot |
| Skill matrix S | `responses_to_skill_matrix()` | % correct per (student, skill) |
| Dot product | `cosine_similarity_observed()` | `np.dot(u, v)` |
| Norm | same | `np.linalg.norm(u)` |
| Cosine similarity | peer matching on S (predictions) and R (demo) | angle between vectors |
| Weighted completion | `predict_missing_scores()` | similarity-weighted peer average |

Latent ability profiles are used **only** to set P(correct) when generating synthetic responses — they are not the displayed score.

## Algorithm

1. Assign ~60% of skills per student for testing (all 3 items per tested skill)
2. Simulate MCQ responses from latent ability + item difficulty
3. **S[i,j] = 100 × (# correct / # attempted)** for tested skills
4. Cosine similarity on observed skill dimensions
5. Priority: `(100 - predicted) × foundational_weight` for untested skills

## Assignment requirements

1. **Non-trivial linear algebra:** R, Q, S matrices; vectors; dot products; norms; cosine similarity; weighted imputation.
2. **Real problem:** Cannot test every SAT skill monthly — surface hidden foundational gaps.
3. **Verifiable artifacts:** Item bank, response log, derived scores, notebook, CSVs, PNGs, dashboard JSON.

## Project structure

```
items.py               # 216-item MCQ bank
skills.py              # 72-skill catalog
lib.py                 # response generation + aggregation + prediction
predict.py             # CLI
export_frontend.py     # dashboard JSON with observedWork
sat_skill_gap_prediction.ipynb
frontend/              # Reading Rooms styled demo UI
data/                  # item_bank, student_responses, skill scores
output/                # recommendations + figures
```
