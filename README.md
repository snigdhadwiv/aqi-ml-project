# AQI Analysis: Regression and Classification on Indian Monitoring Station Data

**Author:** Snigdha Dwivedi | **Roll No:** 2023BTCSE018 | **Course:** Machine Learning (STBT2102)  
**Faculty:** Dr. Dileep Kumar Singh | **University:** Jagran Lakecity University, Bhopal

---

## Project Overview

This project applies machine learning to predict Air Quality Index (AQI) using data from Indian pollution monitoring stations. Two tasks are addressed:

- **Task 1 – Regression:** Predict the numeric AQI value (0–500+)
- **Task 2 – Classification:** Predict the AQI category (Good / Satisfactory / Moderate / Poor / Very Poor / Severe)

**Dataset Source:** [Kaggle – Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)  
**GitHub Repository:** [https://github.com/snigdhadwiv/aqi-ml-project](https://github.com/snigdhadwiv/aqi-ml-project)

---

## Repository Structure

```
aqi-ml-project/
├── Snigdha_2023BTCSE018_MLProject.ipynb   # Main Jupyter notebook
├── Snigdha_2023BTCSE018_MLProject.pdf     # Project report
├── README.md                               # This file
└── data/
    ├── station_day.csv                     # Primary dataset
    ├── stations.csv                        # Station metadata
    ├── city_day.csv                        # City-level daily data
    └── city_hour.csv                       # City-level hourly data
```

---

## Requirements

### Python Version
Python 3.8 or higher

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Or install all at once using:

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
jupyter>=1.0.0
```

---

## How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/snigdhadwiv/aqi-ml-project.git
cd aqi-ml-project
```

### Step 2: Download the Dataset

1. Go to [https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
2. Download and extract the ZIP file
3. Place the following CSV files inside a `data/` folder in the project directory:
   - `station_day.csv`
   - `stations.csv`
   - `city_day.csv`
   - `city_hour.csv`

### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Snigdha_2023BTCSE018_MLProject.ipynb` in your browser.

### Step 4: Run All Cells

In Jupyter: **Kernel → Restart & Run All**

The notebook will automatically:
- Load and merge the datasets
- Preprocess and clean the data
- Generate all EDA visualizations
- Train all regression and classification models
- Print performance metrics and comparison tables

---

## Models Implemented

| Type | Models |
|------|--------|
| Regression | Linear Regression, Ridge, Lasso, Random Forest |
| Classification | Decision Tree, XGBoost |
| Unsupervised | PCA (Principal Component Analysis) |

---

## Key Results

### Regression (Predicting AQI value)

| Model | RMSE | R² Score |
|-------|------|----------|
| **Random Forest** | **37.16** | **0.9200** |
| Linear Regression | 53.39 | 0.8349 |
| Ridge Regression | 53.39 | 0.8349 |
| Lasso Regression | 53.39 | 0.8349 |

### Classification (Predicting AQI Category)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **XGBoost** | **79.06%** | **0.7902** |
| Decision Tree | 73.15% | 0.7292 |

---

## Key Findings

- **PM2.5** is the strongest single predictor of AQI (correlation = 0.81)
- Random Forest outperformed linear models by ~8.5% in R² — confirming non-linear relationships
- XGBoost outperformed Decision Tree by ~6% in accuracy
- Rush hour pollution spikes at **8 AM** and **7 PM** were observed in hourly data
- **Weekend air quality** is slightly cleaner than weekdays
- **Winter months** show significantly higher AQI due to temperature inversions

---

## Notes

- The notebook saves a cleaned dataset (`station_day_cleaned_final.csv`) to disk after preprocessing — this is expected behavior.
- XGBoost label encoding is handled internally; no manual encoding is needed before running.
- Some FutureWarnings from seaborn may appear — these are cosmetic and do not affect results.

---

## Contact

**Snigdha Dwivedi** | JLU ID: JLU08236 | Roll No: 2023BTCSE018  
Jagran School of Engineering, Jagran Lakecity University, Bhopal
