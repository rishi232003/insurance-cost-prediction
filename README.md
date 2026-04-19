# Medical Insurance Cost Prediction

A full end-to-end machine learning project that predicts medical insurance charges using a dataset of 1,338 patients. The goal was to build something that actually works in a real sense, not just fit a model and call it done. I wanted to understand why the model makes the predictions it does, so I layered in an LLM explanation component at the end using Ollama and Llama 3.2.

---

## What This Project Does

It takes patient demographics (age, BMI, smoking status, region, sex, number of children) and predicts their annual medical insurance charges. The interesting part isn't the prediction itself, it's the pipeline that gets there: careful EDA, thoughtful feature engineering, a proper model progression, and finally an LLM layer that translates model output into plain English for a non-technical audience.

---

## Project Structure
├── insurance.csv              # Source dataset (1,338 rows, 7 features)
├── insuranceproj.ipynb # Full notebook with all phases
└── README.md

---

## How I Built It

### Phase 1: EDA

Before touching any models I wanted to understand the data properly. The charges distribution is heavily right-skewed, which immediately pointed to a log transform. A Shapiro-Wilk test confirmed non-normality (stat=0.81, p=7.2e-24). After the log transform the stat jumped to 0.98, much closer to normal.

The correlation heatmap showed smoker_num sitting at 0.79 correlation with charges, which is enormous for real-world data. That single variable explains roughly 62% of the variance on its own. Age (0.30) and BMI (0.20) are meaningful but secondary. Region and sex were basically noise from the start, and the pairplots confirmed it.

### Phase 2: Feature Engineering

This phase did most of the heavy lifting. I one-hot encoded sex and region, log-transformed the target, and built three interaction terms based on what EDA showed:

- `age_x_smoker`: age times smoker status
- `bmi_x_smoker`: BMI times smoker status  
- `obese_x_smoker`: obesity flag (BMI >= 30) times smoker status

The reasoning is that an older smoker or a high-BMI smoker isn't just additive risk, the factors compound. A 60-year-old smoker has a fundamentally different cost profile than either a 60-year-old non-smoker or a young smoker. I wanted to give the models a direct way to capture that relationship rather than hoping they'd find it on their own. The fact that both tree models ranked these interaction terms in the top 3 features later on validated this decision.

Final feature set: 12 features, 80/20 train/test split, StandardScaler applied to training data only.

### Phase 3: Linear Regression Baseline

I ran OLS, Ridge, and Lasso before touching any complex models. All three hit R² of 0.857 and RMSE around 0.358. Ridge and Lasso came out virtually identical to plain OLS, which tells me overfitting was never really a concern here. The dataset is small and clean enough that regularization had almost nothing to do.

The residual plot showed a clear curve, residuals consistently positive at low predicted values and widening spread at the high end. That's leftover nonlinear structure the model couldn't capture, which set up the case for XGBoost.

**Baseline to beat: RMSE 0.3585, R² 0.8571**

### Phase 4: XGBoost

I ran a baseline with hand-picked parameters first, then a GridSearchCV tuned version across 72 parameter combinations with 5-fold cross validation. The baseline edged out the tuned model slightly (R² 0.8669 vs 0.8647, RMSE 0.3460 vs 0.3488). GridSearch optimizes for average CV performance across folds, not a specific test split. With only 268 test rows a difference of 0.003 RMSE is noise. In production I'd trust the CV-tuned model.

The best params were conservative: max_depth of 3, learning_rate of 0.01, subsample and colsample_bytree both at 0.8. Shallow trees with a slow learning rate tells me the signal in this dataset is relatively clean and the model didn't need to go deep.

Feature importance showed smoker_num dominating at around 0.35 gain, with bmi_x_smoker (0.20) and age_x_smoker (0.16) ranking second and third above raw age (0.14). XGBoost can find interactions through tree splits on its own, so it didn't need those engineered features. The fact that it leaned on them anyway is direct validation of the Phase 2 decisions.

**XGBoost result: RMSE 0.3460, R² 0.8669**

### Phase 5: Random Forest

I added Random Forest to sit between linear regression and XGBoost and give the final comparison a cleaner progression. Random Forest builds hundreds of deep independent trees in parallel and averages their predictions (bagging). XGBoost builds trees sequentially where each one corrects the mistakes of the last (boosting). In theory boosting should win on a dataset like this, and it did.

Tuned Random Forest landed at R² 0.8615, RMSE 0.3529, right where I expected. The interesting part was the feature importance comparison. XGBoost ranks smoker_num first by a wide margin. Random Forest ranks age first and drops smoker_num to fourth. Neither answer is wrong, they're just measuring different things. Random Forest scores features by impurity reduction across deep trees on the full dataset. Age has tons of possible split points across all 1,338 rows so it racks up impurity score naturally. XGBoost uses shallow trees focused on fixing errors, and since the biggest errors cluster around smoker cases, it goes straight to smoker_num.

### Phase 6: Model Comparison

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost (baseline) | 0.346 | 0.183 | 0.867 |
| XGBoost (tuned) | 0.349 | 0.189 | 0.865 |
| Random Forest (tuned) | 0.353 | 0.191 | 0.862 |
| OLS / Ridge / Lasso | 0.358 | 0.191 | 0.857 |
| Random Forest (baseline) | 0.372 | 0.193 | 0.846 |

Converting the best model back to dollar space: **RMSE $4,335 and MAE $1,941** on charges ranging from $1,122 to $63,770. The gap between those two numbers tells the real story. RMSE gets inflated by a handful of extreme high-cost cases. MAE is the honest metric here. Being off by an average of $1,941 with only 7 input features and zero medical history or claims data is a reasonable result.

The small gap between XGBoost and linear regression (0.012 RMSE, 0.01 R²) tells me the feature engineering was doing most of the work all along. By the time any of these models saw the data, the log transform, interaction terms, and obese flag had already captured the main structure. The models were competing over the scraps.

### Phase 7: LLM Explanation Layer (Ollama + Llama 3.2)

The last phase adds a local LLM explanation layer using Ollama. After the best model makes a prediction for a patient, the full patient profile, all model predictions, and overall model performance metrics get passed to Llama 3.2 as a structured prompt. The model returns a 4 to 5 sentence plain-English analysis of which model performed best, how accurately each model predicted that specific patient, and which features are likely driving that patient's cost.

This is designed for a stakeholder audience that doesn't care about R² but does care about why a specific patient's charges are estimated at $X. It runs entirely locally, no API keys required.

---

## Results Summary

XGBoost is the right model for this dataset. It's the best performer, handles the nonlinearity the linear model couldn't capture, and is interpretable enough through feature importance to explain which variables are driving predictions. The interaction terms built in Phase 2 showed up as top features in both tree models, which validated that the EDA-driven feature engineering was on the right track.

The remaining prediction errors are concentrated in the $20k to $35k range where smokers and non-smokers start to overlap in ways the model can't always untangle cleanly. More data and richer features (claims history, pre-existing conditions, medication usage) would close that gap. Without them, any model is going to struggle on the high-cost tail regardless of how well it's tuned.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, SciPy, Ollama, Llama 3.2

---

## How to Run

1. Clone the repo and install dependencies
2. Download Ollama and pull the Llama 3.2 model (`ollama pull llama3.2`)
3. Run the notebook top to bottom
4. Phase 7 requires Ollama running locally (`ollama serve`)

