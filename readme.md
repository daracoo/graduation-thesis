# 🏋️ Data-Driven Fitness Recommendation System

A machine learning-based personalized fitness program recommendation system built as a graduation thesis project. The system analyzes user health and behavioral metrics to recommend tailored workout programs from a catalog of 2,597 fitness programs backed by 605,000+ exercise entries.

---

## 🎯 Project Overview

The system combines two datasets to build an end-to-end intelligent recommendation pipeline:
* **Gym Members Dataset** — used for user profiling, fitness level classification, and calorie burn prediction.
* **600K Fitness Exercise Dataset** — used as the program catalog for content-based recommendations.

The system takes a user's physical and behavioral inputs, automatically predicts their fitness level, estimates their calorie burn, and returns the top 5 most suitable workout programs along with a full Week 1 Day 1 exercise breakdown.

---

## 🧠 Machine Learning Pipeline

| Stage | Method | Result |
| :--- | :--- | :--- |
| **Fitness Level Classification** | XGBoost Classifier | 90.26% accuracy, F1 = 0.90 |
| **Calorie Burn Prediction** | XGBoost Regressor | R² = 0.9888, RMSE = 30.51 kcal |
| **User Clustering** | K-Means (K=3) | 3 personas identified |
| **Program Clustering** | K-Means (K=2) | 2 program types identified |
| **Program Recommendation** | Cosine Similarity | 84.7% avg relevance score |

---

## 🔑 Key Findings

1. **Behavior defines fitness level** — Workout frequency and session duration account for ~70% of feature importance in the classifier. Age, BMI, and heart rate contribute very little.
2. **Calorie burn is highly predictable** — Even Linear Regression achieves R²=0.98, confirming a near-linear relationship between session metrics and calorie output.
3. **Three distinct user personas exist** — Elite Athletes (21.5%), Lean Casuals (53.4%), and High-BMI Beginners (25.1%). Average age is ~38 years across all clusters.
4. **Programs split into two structural types** — Long Heavy Programs (12w, ~74min) and Short Light Programs (5w, ~65min) with a clear silhouette peak at K=2.
5. **Goal vocabulary gap** — "Weight Loss" does not exist in the 600K dataset's goal vocabulary, revealing a real-world limitation relevant to production recommendation system design.

---

## 📁 Project Structure

```text
thesis/
│
├── notebook.ipynb              # Main notebook — all 5 phases
│
├── 600k/                       # 600K dataset folder (not included)
│   ├── program_summary.csv
│   └── programs_detailed_boostcamp_kaggle.csv
│
├── members/                    # Gym members dataset folder (not included)
│   └── gym_members_exercise_tracking.csv
│
├── figures/                    # All generated visualizations (13 PNG files)
│   ├── 1_correlation_heatmap.png
│   ├── 2_calories_by_experience.png
│   ├── 3_goal_distribution.png
│   ├── 4_level_goal_heatmap.png
│   ├── 5_progression_over_weeks.png
│   ├── 6_confusion_matrix_classifier.png
│   ├── 7_feature_importance_classifier.png
│   ├── 8_regression_model_comparison.png
│   ├── 9_predicted_vs_actual_calories.png
│   ├── 10_kmeans_optimal_k.png
│   ├── 11b_kmeans_pca_k3.png
│   ├── 12_program_kmeans_optimal_k.png
│   └── 13_program_clusters_pca.png
│
└── models/                     # All saved ML models (generated after running notebook)
    ├── classifier_xgboost.pkl
    ├── regressor_xgboost.pkl
    ├── scaler_classifier.pkl
    ├── scaler_regressor.pkl
    ├── scaler_summary.pkl
    ├── mlb_level.pkl
    ├── mlb_goal.pkl
    ├── kmeans_users.pkl
    ├── kmeans_programs.pkl
    └── cosine_similarity_matrix.pkl
```
---

## 📊 Data

Datasets are not included in this repository to keep it lightweight.

### 📁 Expected folder structure
Place the datasets in the following directories at the project root:

- `600k/`
- `members/`

### ⬇️ Download datasets

- **600K Fitness Exercise Dataset**  
  https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset

- **Gym Members Exercise Dataset**  
  https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset

> 💡 Tip: After downloading, extract the files and place them directly inside the folders above.

### ⚠️ Note
Make sure the folder and file names match the paths used in the notebook (`notebook.ipynb`), otherwise the data loading will fail.

---

## ⚙️ Requirements

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

| Library | Version tested |
|---|---|
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| scikit-learn | ≥ 1.3 |
| xgboost | ≥ 2.0 |
| joblib | ≥ 1.3 |

---

## 🚀 How To Run

### Full Project (all phases)
Open `notebook.ipynb` and run all cells from top to bottom. This will:
- Load and explore both datasets
- Generate all 13 visualizations into `/figures`
- Train all ML models
- Build and evaluate the recommendation engine
- Save all trained models into `/models`

### Recommendation Only (no retraining)
If the `/models` folder already exists from a previous run, load the system directly:

```python
import joblib
from pathlib import Path

models_dir = Path("models")
best_clf   = joblib.load(models_dir / "classifier_xgboost.pkl")
best_reg   = joblib.load(models_dir / "regressor_xgboost.pkl")
# ... load remaining models (see notebook Phase 5 for full loading block)

# Then call:
result = recommend(
    age=28, weight_kg=80.0, height_m=1.78, gender="Male",
    workout_type="Strength", workout_frequency=4, session_duration=1.2,
    fat_percentage=20.0, water_intake=2.5, max_bpm=182, avg_bpm=150,
    resting_bpm=60, goal="Bodybuilding", equipment="Full Gym",
    program_length=12, time_per_workout=75
)
```

---

## 📈 Example Output

### 🧍 User Profile
- **Age:** 28  
- **Gender:** Male  
- **BMI:** 25.2  

### 📊 Predictions
- **Fitness Level:** Intermediate  
- **Estimated Calorie Burn:** 952 kcal/session  

### 🎯 Program Details
- **Goal:** Bodybuilding  
- **Equipment:** Full Gym  
- **Program Length:** 12 weeks  
- **Session Duration:** 75 min  

### 🏆 Top 5 Recommended Programs
1. **[Program Name]** — Similarity: 0.97  
2. **[Program Name]** — Similarity: 0.95  
3. ...  

### 🗓️ Sample Workout Plan
**Week 1 · Day 1**
1. Bench Press (Barbell) — 3 sets × 8 reps  
2. Squat (Barbell) — 4 sets × 6 reps  
3. ...

---

## 🎓 Academic Context

This project was developed as a graduation thesis in the field of Data Science and Machine Learning. It demonstrates the application of supervised learning, unsupervised learning, and content-based filtering in a real-world fitness recommendation context.

**Thesis title:** *"Personalized Fitness Recommendation Through User Profiling and Machine Learning: A Data Science Approach to Workout Program Selection"*

---

## 📄 License

This project is for educational and research purposes only. The datasets used belong to their respective authors on Kaggle. Please respect the original platform's terms of service when using the 600K dataset.