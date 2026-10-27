# Predicting Traffic Impact Severity from Car Accident Data

Machine learning models that predict how severely a car accident disrupts surrounding traffic, built on the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) (7M+ records, 2016-2021).

Random Forests reach **~92% accuracy and 0.97 ROC AUC**, and the most predictive signals turn out to be *when* the accident happened and *what the report says* about it (e.g. whether a lane was "blocked" or "closed").

---

## Motivation

Collisions have a ripple effect on everyone behind them: blocked lanes, closures, and rubbernecking all add delay. If we can identify what makes an accident *traffic-severe*, law enforcement and first responders can respond more effectively, and extra resources can be directed to areas with a high rate of severe incidents.

## Dataset

- **Source:** [US Accidents (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) by Sobhan Moosavi
- **Size:** 7M+ accident records, 45 feature columns, 2016-2021
- **Target:** `Severity`, an integer from 1 (least impact on traffic) to 4 (greatest impact)
- **Class imbalance:** ~79% of records are severity 2; severity 4 is only ~5%
- **Feature groups:** time, location, weather, roadway information, and a free-text `Description`

**Preprocessing**
- Dropped `ID` and `Source` (no predictive value)
- Encoded categorical features as numeric
- Turned `Description` into a **bag-of-words** representation (top 500 words) and concatenated it with the original features
- Due to compute limits, trained on a **random 10% sample** of the data

## Methods

### Random Forest
- Swept forest size (20, 30, 50, 75, 150 trees); performance plateaued between 75 and 150
- Swept `max_features` (10, 15, 25, 35); plateau past 35
- Compared training on the **original (imbalanced)** data vs. data balanced with **SMOTE** (chosen over random over-sampling, which only duplicates minority rows)
- Tested removing low-variance features; no meaningful improvement

### K-Nearest Neighbors
- Normalized features so large-scale features don't dominate distance
- Swept `k` (5, 10, 15, 30); best was `k = 15`
- Evaluated on both imbalanced and SMOTE-balanced data

## Results

### Overall metrics

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Random Forest (75 trees, 35 features), imbalanced | 0.928 | 0.928 | 0.928 | 0.927 | 0.976 |
| Random Forest (75 trees, 35 features), SMOTE | 0.924 | 0.923 | 0.924 | 0.923 | 0.974 |
| KNN (k=15), imbalanced | 0.755 | 0.745 | 0.755 | 0.746 | 0.794 |
| KNN (k=15), SMOTE | 0.663 | 0.740 | 0.663 | 0.686 | 0.774 |

### Per-class recall (confusion matrix diagonals)
"README.md" 123L, 5995B

