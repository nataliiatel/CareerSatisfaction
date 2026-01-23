# Career Satisfaction Prediction

A machine learning project to predict career success and satisfaction using student academic performance, skills, and professional development metrics.

## Project Overview

This project builds a RandomForest classifier to predict career satisfaction outcomes based on educational background, academic performance, and professional engagement metrics. The model is exposed via a FastAPI endpoint for real-time predictions.

### Dataset

- **Primary dataset**: `data/education_career_success_satisfaction.csv` (402 samples)
- **Features**: 18 numeric and categorical variables
  - Academic: GPA (high school, university), SAT Score, Field of Study
  - Professional Development: Internships, Projects, Certifications
  - Skills: Soft Skills Score, Networking Score
  - Career Metrics: Job Offers, Starting Salary, Current Job Level
- **Target**: `target` (binary: 0=low satisfaction, 1=high satisfaction)

### Model

- **Algorithm**: RandomForestClassifier (100 trees, stratified train/test split)
- **Preprocessing**: ColumnTransformer with StandardScaler (numeric) and OneHotEncoder (categorical)
- **Performance** (test set):
  - Accuracy: 1.0
  - ROC AUC: 1.0
  - All classes: precision=1.0, recall=1.0, f1-score=1.0

### Model Evaluation Results

**Classification Report (Test Set)**:
```
Class 0 (Low Satisfaction):
  - Precision: 1.0
  - Recall: 1.0
  - F1-Score: 1.0
  - Support: 54 samples

Class 1 (High Satisfaction):
  - Precision: 1.0
  - Recall: 1.0
  - F1-Score: 1.0
  - Support: 26 samples

Overall Accuracy: 1.0 (80/80 samples correct)
```

**Overfitting Analysis**:

5-Fold Cross-Validation Results:
- Train Accuracy: mean=1.0000, std=0.0000
- Validation Accuracy: mean=0.9950, std=0.0100
- Gap between train and validation: 0.005 (minimal overfitting)

**Interpretation**: The model achieves perfect training accuracy and near-perfect validation accuracy (99.5%), indicating strong generalization across folds. The low standard deviation (0.01) demonstrates consistent performance across all 5 folds.

**Artifacts Generated**:
- `models/roc_auc.png` — ROC curve with AUC=1.0
- `models/learning_curve.png` — learning curve showing train/validation accuracy across sample sizes
- `models/overfit_summary.json` — numerical cross-validation statistics

### Artifacts

After training, the following files are generated in `models/`:
- `model.joblib` — trained pipeline
- `metrics.json` — evaluation metrics (accuracy, classification report, ROC AUC)
- `metadata.json` — feature names for API input alignment
- `roc_auc.png` — ROC curve visualization
- `learning_curve.png` — learning curve (from overfitting evaluator)
- `overfit_summary.json` — cross-validation train/test scores
- `correlations.json` — numeric feature correlations with target
- `categorical_target_means.json` — categorical feature analysis

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Model

```bash
# Default: uses education_career_success_satisfaction.csv
./.venv/bin/python -m src.train --debug

# With explicit target column name
./.venv/bin/python -m src.train --target target

# With explicit data file
./.venv/bin/python -m src.train --data data/education_career_success_satisfaction.csv
```

### 3. Run API Server

```bash
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Server will be available at `http://127.0.0.1:8000`. OpenAPI docs at `http://127.0.0.1:8000/docs`.

### 4. Make Predictions

Example request (student S649):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "S649",
    "age": 21,
    "gender": "Male",
    "high_school_gpa": 3.9,
    "sat_score": 1580,
    "university_gpa": 3.9,
    "field_of_study": "Computer Science",
    "internships_completed": 4,
    "projects_completed": 9,
    "certifications": 5,
    "soft_skills_score": 8,
    "networking_score": 9,
    "job_offers": 0,
    "starting_salary": 148000,
    "years_to_promotion": 1,
    "current_job_level": "Senior",
    "work_life_balance": 5,
    "entrepreneurship": "No"
  }'
```

Expected response:
```json
{
  "prediction": 1,
  "probability": 0.95
}
```

## Project Structure

```
.
├── src/
│   ├── data_loader.py       # CSV loading, preprocessing, target detection
│   ├── train.py             # Model training, evaluation, artifact saving
│   └── evaluate_overfit.py  # Cross-validation & learning curve analysis
├── app/
│   └── main.py              # FastAPI app with /predict endpoint
├── data/
│   ├── sample.csv           # Synthetic sample dataset
│   └── education_career_success_satisfaction.csv  # Main dataset
├── models/                  # Generated after training
│   ├── model.joblib
│   ├── metrics.json
│   ├── metadata.json
│   ├── roc_auc.png
│   └── ...
├── tests/
│   └── test_train.py        # Unit tests for training pipeline
├── notebooks/
│   └── demo.ipynb           # Jupyter notebook demo
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # MIT License
```

## Data Loading & Target Detection

The `src/data_loader.py` module includes intelligent target column detection:
- Automatically detects `target` column (case-insensitive)
- Falls back to common synonyms: `outcome`, `success`, `label`, `y`, `satisfaction`
- Handles CSV files with title rows (skips first row if needed)
- Accepts explicit `--target` flag if ambiguous

Example:
```python
from src.data_loader import load_data
X, y = load_data('data/education_career_success_satisfaction.csv', target_cols='target')
```

## Evaluation & Overfitting Analysis

Run the overfitting evaluator to assess model generalization:

```bash
./.venv/bin/python -m src.evaluate_overfit --cv 5
```

This generates:
- Cross-validation train/test scores (with train_score = 1.0, val_score ≈ 0.995)
- Learning curve plot (`models/learning_curve.png`)
- Summary statistics (`models/overfit_summary.json`)

## Feature Correlation Analysis

Compute and save feature correlations with the target:

```bash
# Via Python snippet
./.venv/bin/python - <<'PY'
import pandas as pd, json
from src.data_loader import load_data

X, y = load_data('data/education_career_success_satisfaction.csv')
df = pd.concat([X, y.rename('target')], axis=1)

# Numeric correlations
corr = df.select_dtypes(include=['number']).corr()['target'].sort_values(key=lambda s: s.abs(), ascending=False)
print(corr.head(15))

with open('models/correlations.json', 'w') as f:
    json.dump(corr.to_dict(), f, indent=2)
PY
```

**Top Correlations**:
- `Starting_Salary`: 0.811
- `Job_Offers`: 0.808
- `SAT_Score`: 0.798
- `Soft_Skills_Score`: 0.797
- `University_GPA`: 0.786

## Testing

Run unit tests:

```bash
./.venv/bin/python -m pytest tests/ -v
```

## Dependencies

- pandas, numpy — data manipulation
- scikit-learn — model training & evaluation
- FastAPI, uvicorn — API server
- joblib — model serialization
- matplotlib — plotting (ROC curves, learning curves)
- pydantic — request validation

See `requirements.txt` for versions.

## License

MIT License — see `LICENSE` file.
