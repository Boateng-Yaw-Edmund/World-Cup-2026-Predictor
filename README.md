# ⚽ FIFA World Cup 2026 Predictive Analytics & Tournament Simulation Pipeline

An end-to-end machine learning and data engineering pipeline designed to predict international football match outcomes and simulate the complete, expanded 48-team **FIFA World Cup 2026** tournament framework. 

This architecture blends long-term team skill vectors (historical international Elo ratings) with high-frequency, short-term momentum data (rolling time-series form tracking) to map multi-class structural features to match outcomes (Home Win, Draw, Away Win).

---

## 🛠️ Tech Stack & Frameworks
* **Data Engineering & Vectorization:** `Pandas`, `NumPy`, `Itertools`
* **Machine Learning Pipelines:** `Scikit-Learn` (Ensembles, Preprocessing, Metrics)
* **Gradient Boosting Frameworks:** `XGBoost` (eXtreme Gradient Boosting Classifier)
* **Statistical Diagnostics & Data Viz:** `Seaborn`, `Matplotlib`

---

## 📊 Datasets & Ingestion Sources

The pipeline ingests data across multiple structural historical layers. The foundational underlying data files can be compiled from the following open-source football analytics repositories:

1. **International Football Results (1872 - Present):** [Kaggle Dataset by Mart Jürisoo](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
   * Ingests chronological records of historical match metrics, scores, tournaments, locations, and neutral site status markers (`results.csv` and `shootouts.csv`).
2. **World Football Elo Ratings:** [Kaggle Dataset by Ernest Wonyaya](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)
   * Tracks time-series data of global relative team skill levels and chronological Elo point shifts over time (`eloratings.csv`).

---

## ⚙️ Core Engineering & Architecture Highlights

### 1. Temporal Integrity & Look-Ahead Bias Prevention
To predict chronological events without contaminating the training data, team-level momentum features (`form_5`) are engineered using a strict chronological sort and a `.shift()` window constraint. This ensures that a fixture evaluated at time $t$ only leverages performance vectors native to the historical timeline up to $t-1$. The feature space is completely sealed against future look-ahead data leakage.

### 2. Generalization via 5-Fold Cross-Validation
Rather than relying on volatile, single train-test splits that create overly optimistic performance metrics, the pipeline executes a full 5-Fold Cross-Validation across the entire historical data footprint. This stress-tests the algorithmic architectures across multiple data shuffles, yielding an honest, robust baseline for operational model selection.

### 3. Out-of-Sample Regulation-Compliant Simulation Engine
The simulation framework acts as a pure, out-of-sample production inference layer. The script ingests the official 48-team group stage configurations, calculates round-robin points tables, dynamically processes FIFA's complex tie-breaking regulations to extract the top 8 "Best 3rd Place" teams, and pipes them down a strict, official knockout bracket tree to determine the world champion.

### 4. Interactive Group Stage Standings
The pipeline includes logic to display the final group stage standings upon the conclusion of the group stage matches. This provides a clear view of team points, goal differences, and qualification status for the knockout rounds before proceeding to the Round of 32.

---

## 📊 Model Evaluation & Deep-Dive Trade-offs

### The Train-Test Split Trap vs. Cross-Validation Truth
During initial prototyping on an isolated 80/20 partition, the models showed conflicting signals. However, subjecting the pipeline to a 5-Fold Cross-Validation stress test revealed the true performance metrics:

| Predictive Model Configuration | Single Holdout Split Score | 5-Fold Cross-Validation Mean | Model Generalization Status |
| :--- | :---: | :---: | :--- |
| **XGBoost Classifier** | 54.19% | **49.99% (+/- 2.43%)** | Overfitted on Partition Noise |
| **Random Forest Ensemble** | 52.51% | **53.03% (+/- 2.32%)** | **Operational Champion (Stable)** |

### Algorithmic Insight
Because the feature space is optimized and highly focused (6 dimensions including Elo deltas and lagged form weights), the gradient boosting mechanisms within **XGBoost** over-indexed on localized variance, causing its accuracy to collapse to a coin-flip (~50%) across unseen folds. 

Conversely, the **Random Forest** classifier's bootstrap aggregation (bagging) framework stabilized predictions by averaging uncorrelated decision trees. This design choice provided a crucial mathematical shock absorber against the high-entropy anomalies and historical upsets native to international sports analytics.

### Error Analysis via Confusion Matrices
Multi-class confusion matrices generated via `Seaborn` highlighted a standard domain obstacle in football analytics: the **Draw Penalty**. 

While both models displayed exceptional precision when isolating clean Home Wins (benefiting from historical home-field advantage metrics) and clear Away Wins, the models frequently misclassified actual draws as Home Wins. This is because draws in international football represent high-entropy chaos (e.g., late equalizers, red cards, missed penalties) that mathematical features alone struggle to resolve, establishing a natural performance ceiling right around the ~53% boundary.

---

## 📈 Tournament Bracket Handoff Simulation

The operational Random Forest model was deployed to run a fully deterministic simulation of the upcoming 2026 World Cup tree. 

Rather than relying on generic flat lists, the knockout engine utilizes a dynamic positional dictionary tracking true bracket connectivity constraints. It maps official group positions (e.g., Runner-up Group A vs. Runner-up Group B) and match index keying down to the Grand Final, resolving high-stakes knockout draws by analyzing raw probability distributions (`predict_proba`) and historical Elo deadlocks to simulate penalty shootouts.

---

## 🏆 Simulation Results & Findings

Based on our final Random Forest tournament simulation, here is how the 2026 World Cup bracket unfolded, highlighting the predicted narratives and upsets:

### 👑 The Projected 2026 Champion
* **Winner:** France 🇫🇷
* **Runner-Up:** Spain 🇪🇸
* **Story of the Tournament:** In a clash of European titans, France edges out Spain in the Grand Final to claim the 2026 World Cup title, solidifying their dominance in international football after a grueling knockout run that includes defeating Germany, Brazil, and Spain in succession.

### 🤯 Major Knockout Upsets
* **The Giant Killers:** The most stunning result comes in the Round of 32, where tournament favorites **Belgium** are shockingly knocked out by **Cabo Verde**.
* **Nordic Upset:** **Norway** pulls off an incredible Round of 16 victory against a stacked **Portugal** squad, cementing one of the biggest upsets of the knockout stage.
* **Defending Champions Fall Short:** Defending champions **Argentina** make a strong run but are ultimately halted in the Semi-Finals by Spain.

### 🐎 Dark Horses & Cinderella Stories
* **Surprise Quarter-Finalists:** **Iran** makes an unprecedented deep run. After advancing as a 3rd place team from Group G, they defeat Group D winners Paraguay in the Round of 32, followed by Cabo Verde in the Round of 16, before finally falling to Spain in the Quarter-Finals.
* **Morocco Strikes Again:** Proving 2022 was no fluke, **Morocco** reaches the Quarter-Finals once again, dispatching Japan and Bosnia and Herzegovina before being eliminated by the eventual champions, France.
* **Best 3rd Place Chaos:** The expanded 48-team format proves highly volatile. Several 3rd place teams cause absolute chaos—most notably **Cabo Verde**, **Iran**, and **Norway**, who all advance as 3rd place teams and win high-stakes knockout matches against group winners.

---

## 🚀 Pipeline Directory Structure
```text
├── index.html              # Browser UI entry point
├── styles.css              # Web UI styling
├── app.js                  # In-browser tournament simulation logic
├── data/
│   ├── results.csv          # Comprehensive historical international fixtures
│   ├── shootouts.csv        # Historic penalty shootout outcome vectors
│   ├── eloratings.csv       # Chronological international Elo points dataset
│   └── former_names.csv     # Geopolitical country name mapping configuration
├── wc2026.ipynb             # Integrated prototyping, engineering & validation notebook
└── README.md                # Project documentation
```

## 🌐 Web UI

The project now includes a browser-based UI, separate from Streamlit and Jupyter.

Start the backend:

```bash
python backend/server.py
```

Then visit:

```text
http://127.0.0.1:8768/
```

## 🔌 Backend API

Install the ML dependencies:

```bash
pip install -r requirements.txt
```

Add the training files expected by the notebook:

```text
data/results.csv
data/shootouts.csv
data/eloratings.csv
```

Start the API:

```bash
python -m backend.server
```

The UI calls:

```text
GET  http://127.0.0.1:8768/api/status
POST http://127.0.0.1:8768/api/train
POST http://127.0.0.1:8768/api/predict
POST http://127.0.0.1:8768/api/simulate
POST http://127.0.0.1:8768/api/compare
```

The backend extracts the notebook pipeline into reusable code: historical result
cleaning, shootout-aware target labels, as-of Elo joins, rolling five-match form,
Random Forest training, optional XGBoost training, and full 2026 tournament
simulation through the 32-team knockout bracket.

Trained models are cached under `backend/cache/` and automatically invalidated
when the source CSV files change. The Random Forest cache is used by default for
fast API restarts; XGBoost is cached separately the first time that model is run.

The UI also includes a model comparison view that runs Random Forest and XGBoost
side by side, showing champion/final differences plus group and knockout
disagreements.
