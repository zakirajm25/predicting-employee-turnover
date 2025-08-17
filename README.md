# Predictive Analytics for Employee Turnover at Salifort Motors: Data-Driven HR Insights

**Short Description:** Using regression and tree‑based machine learning models to analyze HR data, predict employee turnover, and provide actionable recommendations for improving retention.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Business Problem & Objectives](#business-problem--objectives)
3. [Data Understanding](#data-understanding)
4. [Modeling and Evaluation](#modeling-and-evaluation)
5. [Key Questions Answered](#key-questions-answered)
6. [Conclusions & Recommendations](#conclusions--recommendations)
7. [Ethical Considerations](#ethical-considerations)
8. [Tools and Libraries](#tools-and-libraries)
9. [Project Reproducibility](#project-reproducibility)
10. [License](#license)

---

## 1. Project Overview
This capstone project focuses on employee attrition prediction for **Salifort Motors**, a large consulting firm. By combining exploratory data analysis (EDA) with both regression and tree‑based machine learning models, the project seeks to:
- Identify the main factors driving turnover.
- Build predictive models to flag at‑risk employees.
- Provide HR with actionable insights to improve employee satisfaction and retention.

The final outputs include:
- **A one‑page executive summary** for external stakeholders.
- **A complete annotated Jupyter Notebook** containing code, analysis, visualizations, model evaluation, and ethical considerations.

---

## 2. Business Problem & Objectives

**Stakeholder:** Salifort Motors HR Department

**Business Problem:**  
The HR department aims to understand and reduce employee turnover by leveraging available workforce data. They want to know:  
> *What are the key factors that indicate an employee is likely to leave?*

**Objectives:**
- Analyze employee data for patterns linked to attrition.
- Develop and compare regression and tree‑based classification models.
- Select a champion model based on multiple performance metrics.
- Translate results into practical HR recommendations.

---

## 3. Data Understanding

**Data Source:** Internal HR dataset (`HR_dataset.csv`).

**Key Features:**
- Satisfaction level
- Last evaluation score
- Number of projects
- Average monthly hours
- Tenure
- Work accident history
- Left (target variable)
- Promotion in last 5 years
- Department
- Salary category

**Preparation Steps:**
- Standardized column names.
- Verified no missing values.
- Removed duplicate records.
- Retained tenure outliers for model robustness.

**EDA Highlights:**
- Overall attrition rate was moderate, with distinct workload and satisfaction patterns among those who left.
- Employees with extreme workloads (very high or low hours/projects) showed higher attrition.
- Lower satisfaction was a consistent predictor of leaving.

---

## 4. Modeling and Evaluation

**Approach:**
- Models used: Logistic Regression (baseline), Decision Tree, Random Forest, and XGBoost.
- Encoded categorical variables.
- Train/test split: 75%/25%, stratified on target.

**Evaluation Metrics:**
- Accuracy
- Precision, Recall, F1‑score (for “left” class)
- ROC‑AUC

**Process:**
- Established a simple regression baseline.
- Implemented tree‑based models to capture non‑linear relationships.
- Selected champion model based on balanced performance, with priority on correctly identifying at‑risk employees (high recall for “left”).

---

## 5. Key Questions Answered
1. **What drives attrition?**  
   – Low satisfaction, extreme workloads, and certain tenure points correlate strongly with turnover.
2. **Can attrition be predicted accurately?**  
   – Yes; tree‑based models, particularly tuned ensembles, offered strong predictive performance.
3. **Which employees are most at risk?**  
   – Those with low satisfaction scores, high monthly hours, high or very low project counts, and in certain departments or tenure bands.

---

## 6. Conclusions & Recommendations

**Conclusions:**
- Multiple interpretable factors contribute to attrition.
- Predictive modeling can meaningfully flag at‑risk employees.

**Recommendations:**
1. Monitor workloads and ensure balanced project allocations.
2. Conduct targeted engagement for low‑satisfaction employees and those at key tenure milestones.
3. Review recognition and promotion processes to retain high performers.

---

## 7. Ethical Considerations
- **Fairness:** Audit models for bias across demographics.
- **Transparency:** Share general drivers with staff without revealing individual predictions.
- **Supportive Use:** Apply predictions to offer support, not punitive measures.
- **Data Privacy:** Comply with applicable privacy laws; secure sensitive employee data.

---

## 8. Tools and Libraries
- **Python 3.x**
- pandas, NumPy
- scikit‑learn
- matplotlib, seaborn
- XGBoost

---

## 9. Project Reproducibility
1. Clone the repository.
2. Install dependencies from `requirements.txt`.
3. Place `HR_capstone_dataset.csv` in the specified `data/raw/` folder.
4. Open and run the provided Jupyter Notebook in your environment.

---

## 10. License
This project is licensed under the MIT License. See the `LICENSE` file for details.
