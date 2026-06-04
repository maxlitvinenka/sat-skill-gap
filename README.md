# Foundational SAT Skill Gap Prediction

**Standalone prototype — synthetic data only.** Not connected to the Reading Rooms production codebase or database.

Predicts which SAT Reading/Writing skills a student is likely weak in, even when those skills have not been tested yet. Uses **linear algebra**: student-skill matrix, cosine similarity (dot products and norms), and weighted imputation of missing scores.

## Quick start

### Python pipeline

```bash
cd sat-skill-gap-prototype
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python predict.py --export-frontend
```

This generates:
- `data/skill_metadata.csv` — 72 skills with category, level, foundational weight
- `data/synthetic_student_scores.csv` — simulated scores (long format)
- `output/recommendations_*.csv` — ranked gap recommendations
- `output/figures/*.png` — heatmap and bar charts
- `frontend/public/data/dashboard.json` — data for the web UI

Open `sat_skill_gap_prediction.ipynb` for the full linear-algebra walkthrough.

### Localhost dashboard

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

Visual design copied from Reading Rooms (library green, antique gold, warm paper, Cinzel headings, shadcn-style components). **No runtime dependency** on the Reading Rooms repo.

## Algorithm

1. Build student-skill matrix S (75 x 72) from synthetic latent profiles
2. Hide ~40% of scores per student (untested skills)
3. For target student u, compute cosine similarity with peers on shared observed skills
4. Predict missing skills: similarity-weighted average of peer scores
5. Priority: (100 - predicted) x foundational_weight

## Linear algebra (where / what / how)

| Concept | Where | How |
|---------|-------|-----|
| Matrix | lib.py scores array (75, 72) | Rows = student vectors |
| Dot product | cosine_similarity_observed() | np.dot(u, v) |
| Norm | same function | np.linalg.norm(u) |
| Cosine similarity | peer matching | angle between partial skill vectors |
| Weighted completion | predict_missing_scores() | similarity-weighted peer average |

## Assignment requirements

1. **Non-trivial linear algebra:** Students modeled as vectors; data as a matrix; cosine similarity via dot products and norms; predictions via weighted vector comparison.
2. **Real problem:** Tutoring platforms cannot test every SAT skill each month — this prioritizes likely hidden foundational weaknesses.
3. **Verifiable artifacts:** Python code, notebook, CSVs, PNGs, dashboard.json, localhost UI, and this README.

## Design kit provenance

Copied once from Reading Rooms for visual consistency:
- CSS tokens (index.css pattern)
- Tailwind brand colors
- shadcn/ui component patterns
- Recharts theme (chart-theme.tsx)

## Project structure

```
skills.py              # 72-skill catalog
lib.py                 # generation + cosine similarity + ranking
predict.py             # CLI
export_frontend.py     # JSON for React dashboard
sat_skill_gap_prediction.ipynb
frontend/              # Vite + React demo UI
data/                  # CSV outputs
output/                # recommendations + figures
```
