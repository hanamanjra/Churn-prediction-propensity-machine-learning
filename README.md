# Churn-prediction-propensity-machine-learning

# Customer Churn Analysis

This project performs a complete **Exploratory Data Analysis (EDA)** and **predictive modelling** pipeline for understanding and predicting **customer churn**. It aims to provide actionable insights for reducing churn through diagnostic analytics, machine learning, and strategic business recommendations.

---

## Project Objectives

- **Understand** why customers churn.
- **Identify** key trends and relationships influencing churn.
- **Predict** which customers are likely to churn using ML models.
- **Strategise** business solutions to improve retention.

---

## Tech Stack

| Category              | Tools & Libraries                              |
|-----------------------|-------------------------------------------------|
| Data Processing       | `pandas`, `numpy`                              |
| Visualization         | `matplotlib`, `seaborn`                        |
| Modelling             | `scikit-learn`, `MLflow`                       |
| Deployment            | `FastAPI`, `Uvicorn`                           |
| Agile Management      | Daily standups, retros, and MoSCoW prioritisation |

---

## Pre-processing Pipeline

1. **Data type conversion**
2. **Remove duplicates and missing values**
3. **Normality checks**
4. **One-hot encoding**
5. **Z-score normalization**
6. **Outlier removal (univariate & multivariate)**
7. **Multicollinearity check & removal**

---

## EDA Highlights

### Univariate Insights
- **High churn** observed in customers with **higher monthly charges** and **shorter tenure**.
- Recommendation: Introduce **pay-as-you-go** models or **value-based nudges**.

### Bivariate Insights
- **Contract type**, **online security**, and **tech support** correlate negatively with churn.
- **Senior citizens**, **fiber internet users**, and **high charge customers** churn more.
- Solution: Bundle offers, longer contracts, and improved service for fiber customers.

### Grouped Insights
- Customers with **multiple tech/security services** have **lower churn rates**.
- Recommendation: Offer **bundle discounts**, **AI support**, and **free trials**.

---

## Domain-Specific Clustering

- **K-Means Clustering** revealed **4 customer segments** based on tech service adoption.
- **Elbow plot** confirmed optimal cluster number.
- Strategy: **Segment-based targeting** to tailor retention campaigns.

---

## Predictive Modelling & Deployment

- **PCA + KNN model** used to address dimensionality issues and improve performance.
- **MLflow** for experiment tracking and model monitoring.
- **FastAPI + Uvicorn** for REST API deployment of the model.
  - Supports single & batch entry (planned).
  - Future: Add health checks & logging for robustness.

---

## Future Improvements

1. **Enhance model deployment**: Add health check endpoints, logging, monitoring.
2. **Deeper segmentation**: Study demographics of tech adopter clusters.
3. **Extend business analysis**: Integrate CLV, revenue impact, or retention curves.

---

## Stakeholder Focus

Designed with ongoing **stakeholder engagement**:
- Daily sprints & iterations.
- Aligned business questions with data insights.
- Strategic recommendations to improve churn KPIs.

---
