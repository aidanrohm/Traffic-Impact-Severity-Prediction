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

| Model | Sev. 1 | Sev. 2 | Sev. 3 | Sev. 4 |
|---|---|---|---|---|
| Random Forest, imbalanced | 0.75 | 0.96 | 0.88 | 0.56 |
| Random Forest, SMOTE | **0.85** | 0.95 | 0.88 | **0.65** |
| KNN, imbalanced | 0.35 | 0.86 | 0.55 | 0.09 |
| KNN, SMOTE | 0.82 | 0.65 | 0.69 | 0.40 |

### Key takeaways

- **Random Forests far outperform KNN.** KNN likely suffers from the curse of dimensionality (500 bag-of-words features) and from noisy synthetic samples created by SMOTE.
- **SMOTE trades a little overall accuracy for much better minority-class detection.** The balanced forest is ~0.4 points lower on aggregate metrics but detects the least severe and most severe accidents 10 and 9 points better, respectively.
- **Top features** (Random Forest importance): `Start_Time`, the words *"closed"* and *"crash"*, `Weather_Timestamp`, `End_Time`, `Distance(mi)`, *"exit"*, *"road"*, *"accident"*, and *"blocked"*. Several of the most important features come from the free-text description, which captures information (like lane blockage) that structured columns miss.

## Tech Stack

Python, scikit-learn, imbalanced-learn (SMOTE), pandas, NumPy, Matplotlib

## Limitations and Future Work

- Training used only a 10% sample and a 500-word vocabulary because of memory and compute limits; a more powerful machine could support the full dataset, larger forests, and a richer text representation.
- KNN performance could be improved with dimensionality reduction (e.g. PCA) before distance computation.
- Other approaches worth exploring: gradient boosting, TF-IDF or embedding-based text features, and evaluating on a time-based train/test split.

## Citation and Acknowledgments

This project uses the US Accidents dataset. Please cite the dataset creators if you use it:

- Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (2019). *A Countrywide Traffic Accident Dataset.* [arXiv:1906.05409](https://arxiv.org/abs/1906.05409)
- Moosavi, S., Samavatian, M. H., Parthasarathy, S., Teodorescu, R., & Ramnath, R. (2019). *Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights.* [arXiv:1909.09638](https://arxiv.org/abs/1909.09638)

Thanks to the scikit-learn team for the algorithms that made this project possible.

## Author

**Aidan Rohm**, M.S. Computer Science, Fordham University
