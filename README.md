# Predictive Analytics for Employee Attrition at Salifort Motors: A Data-Driven Approach to Enhance Retention

**Short Description:** Leveraging various machine learning models to analyze employee data, predict turnover, and provide actionable HR insights for Salifort Motors.

---

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Business Problem & Objectives](#business-problem--objectives)
3.  [Data Understanding](#data-understanding)
    * [Data Source & Description](#data-source--description)
    * [Dataset Features](#dataset-features)
    * [Data Cleaning & Preparation](#data-cleaning--preparation)
    * [Key EDA Insights & Visualizations](#key-eda-insights--visualizations)
4.  [Modeling and Evaluation](#modeling-and-evaluation)
    * [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
    * [Model Selection & Rationale](#model-selection--rationale)
    * [Evaluation Strategy & Metrics](#evaluation-strategy--metrics)
    * [Model Performance & Champion Model](#model-performance--champion-model)
5.  [Key Questions Answered](#key-questions-answered)
6.  [Conclusions & Actionable Recommendations](#conclusions--actionable-recommendations)
    * [Key Conclusions](#key-conclusions)
    * [Actionable Recommendations for Salifort Motors HR](#actionable-recommendations-for-salifort-motors-hr)
    * [Limitations](#limitations)
    * [Future Work](#future-work)
7.  [Ethical Considerations](#ethical-considerations)
8.  [Tools and Libraries Used](#tools-and-libraries-used)
9.  [Project Reproducibility](#project-reproducibility)
10. [License](#license)

---

## 1. Project Overview
This project addresses the critical issue of employee attrition at Salifort Motors by applying data science methodologies. Through in-depth exploratory data analysis (EDA) and the development of various predictive machine learning models—including logistic regression, decision trees, random forests, and XGBoost—this initiative aims to uncover the primary drivers of employee turnover. The ultimate goal is to equip Salifort Motors' HR department with a robust model capable of identifying employees at risk of leaving, thereby enabling targeted interventions to improve satisfaction and retention. *(Placeholder: The final champion model, [Specify Champion Model Name], achieved [mention key metric, e.g., X% accuracy or Y% F1-score] in predicting attrition.)*

---

## 2. Business Problem & Objectives

**Stakeholder:** HR Department, Salifort Motors

**Business Problem:**
The HR department at Salifort Motors is tasked with enhancing employee satisfaction and reducing attrition rates. While they have collected substantial employee data, they require analytical expertise to extract meaningful insights and address the core question: **"What are the key factors that indicate an employee is likely to leave the company?"**

**Project Objectives:**
* Conduct a comprehensive analysis of the provided HR dataset to identify significant patterns and correlates of employee attrition.
* Develop and evaluate a suite of machine learning models to accurately predict the likelihood of an employee leaving.
* Select a 'champion' model based on clearly defined performance metrics and interpretability.
* Translate model findings and EDA insights into actionable, data-driven recommendations for the HR department to strategically improve employee retention and workplace satisfaction, thereby mitigating the considerable costs associated with employee turnover.

---

## 3. Data Understanding

### Data Source & Description
The analysis is based on the `HR_capstone_dataset.csv` file, which contains employee-related data from Salifort Motors. The original dataset comprised 15,000 records and 10 features. After cleaning, the dataset used for analysis comprised 11,991 unique employee records.

### Dataset Features
The following table details the features available for analysis (original names are provided if they differed from the cleaned, renamed features used in the project):

| Feature Original Name  | Renamed Feature         | Description                                                        | Data Type   |
| :--------------------- | :---------------------- | :----------------------------------------------------------------- | :---------- |
| `satisfaction_level`   | `satisfaction_level`    | Employee-reported job satisfaction level [0–1]                     | `float64`   |
| `last_evaluation`      | `last_evaluation`       | Score of employee's last performance review [0–1]                | `float64`   |
| `number_project`       | `number_project`        | Number of projects employee contributes to                         | `int64`     |
| `average_montly_hours` | `average_monthly_hours` | Average number of hours employee worked per month                  | `int64`     |
| `time_spend_company`   | `tenure`                | How long the employee has been with the company (years)            | `int64`     |
| `Work_accident`        | `work_accident`         | Whether or not the employee experienced an accident while at work (0=No, 1=Yes) | `int64`     |
| `left`                 | `left`                  | Whether or not the employee left the company (Target Variable: 0=No, 1=Yes) | `int64`     |
| `promotion_last_5years`| `promotion_last_5years` | Whether or not the employee was promoted in the last 5 years (0=No, 1=Yes) | `int64`     |
| `Department`           | `department`            | The employee's department                                          | `object`    |
| `salary`               | `salary`                | The employee's salary (Categorical: low, medium, high)             | `object`    |

### Data Cleaning & Preparation
Initial data examination and cleaning involved:
* **Column Renaming:** Standardized column names to `snake_case` (e.g., `average_montly_hours` to `average_monthly_hours`, `Work_accident` to `work_accident`, `time_spend_company` to `tenure`).
* **Missing Value Assessment:** The dataset was confirmed to have no missing values.
* **Duplicate Removal:** Identified and removed 3,008 duplicate entries, resulting in a refined dataset of 11,991 unique records used for analysis.
* **Outlier Identification:** The `tenure` variable exhibited outliers (824 records with tenure outside 1.5-5.5 years based on IQR). The strategy for addressing these outliers was to retain them, understanding that tree-based models are generally robust to outliers, but their impact would be monitored.

### Key EDA Insights & Visualizations
*(Placeholder: This is a critical section. Expand with specific, quantified insights and embed 2-3 key visualizations that tell a story about the data. Examples:)*
* **Attrition Profile:** The cleaned dataset revealed an overall attrition rate of approximately 16.6% (1,991 out of 11,991 employees).
    ```markdown
    ![Distribution of Employees Who Left vs. Stayed](images/left_vs_stayed_ratio.png)
    ```
* **Workload Indicators:** A strong correlation was observed between `average_monthly_hours`, `number_project`, and attrition.
    * *(Placeholder: Specific insight, e.g., "Employees working on 7 projects had a 100% attrition rate, while those with 3-4 projects showed higher retention.")*
    * *(Placeholder: Specific insight, e.g., "Employees who left were often found in two groups: those working significantly fewer hours (avg. X hrs) or significantly more hours (avg. Y hrs) than their peers.")*
    ```markdown
    ![Monthly Hours by Number of Projects, Segmented by Attrition](images/monthly_hours_by_projects.png)
    ```
* **Satisfaction & Tenure Dynamics:**
    * Employees who left had a mean satisfaction score of 0.44 (median 0.41), compared to 0.67 (median 0.69) for those who stayed.
    * *(Placeholder: Specific insight about tenure, e.g., "A notable dip in satisfaction and spike in attrition occurred around the 4-year tenure mark for employees who left.")*
    ```markdown
    ![Satisfaction Level by Tenure, Segmented by Attrition](images/satisfaction_by_tenure.png)
    ```
* *(Placeholder: Other significant findings, e.g., from `salary` or `last_evaluation` distributions. For each, provide a brief textual insight and the Markdown for the embedded image.)*

---

## 4. Modeling and Evaluation

This section details the process of building and evaluating predictive models for employee attrition.

### Feature Engineering & Preprocessing
* **Encoding Categorical Variables:** Categorical features `department` and `salary` were transformed into numerical format using one-hot encoding (`pd.get_dummies`), with `drop_first=True` to prevent multicollinearity.
* **Train-Test Split:** The dataset (with features `X` and target `y = df1['left']`) was partitioned into 75% for training and 25% for testing. The split was stratified based on the target variable `left` to maintain class proportions, and a `random_state=42` was used for reproducibility.
*(Placeholder: Mention if any numerical scaling (e.g., StandardScaler, MinMaxScaler) was applied, though often not strictly necessary for tree-based models which formed the core of the modeling.)*

### Model Selection & Rationale
Based on the project's objective to predict a binary outcome, the following classification models were implemented and evaluated:
1.  **Logistic Regression:** Chosen as a baseline model for its simplicity and interpretability.
2.  **Decision Tree Classifier:** Selected to capture non-linear relationships and provide easily understandable decision rules.
3.  **Random Forest Classifier:** An ensemble method utilized for its robustness against overfitting and improved predictive accuracy over single decision trees.
4.  **XGBoost Classifier:** A powerful and efficient gradient boosting algorithm, often yielding state-of-the-art performance on structured data.

### Evaluation Strategy & Metrics
Model performance was primarily assessed using the following metrics, with a focus on correctly identifying employees likely to leave (the positive class, `left=1`):
* **Accuracy:** Overall correctness of predictions.
* **Precision (for class 1):** Of those predicted to leave, how many actually left.
* **Recall (Sensitivity for class 1):** Of those who actually left, how many were correctly identified. (This was a key metric to minimize missed at-risk employees).
* **F1-Score (for class 1):** The harmonic mean of Precision and Recall, providing a balanced measure.
* **ROC AUC Score:** Measures the model's ability to distinguish between the two classes.
* **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.

Hyperparameter tuning was performed using `GridSearchCV` with cross-validation on the training data to optimize the selected models.

### Model Performance & Champion Model
*(Placeholder:
* Present a clear table summarizing the performance of all tested models (including tuned versions) on the test set, using the metrics above.
* Clearly state which model was chosen as the "champion model" (e.g., Tuned Random Forest or Tuned XGBoost) and provide a strong justification based on its performance (especially recall and F1-score for the 'left' class) and overall suitability for the business problem.
* Discuss insights from the champion model, particularly feature importances.)*

```markdown
*(Placeholder: Example of how you might structure the model performance table - use Markdown table syntax)*
| Model                  | Accuracy | Precision (Left) | Recall (Left) | F1-Score (Left) | ROC AUC |
| :--------------------- | :------- | :--------------- | :------------ | :-------------- | :------ |
| Logistic Regression    | *value* | *value* | *value* | *value* | *value* |
| Decision Tree (tuned)  | *value* | *value* | *value* | *value* | *value* |
| Random Forest (tuned)  | *value* | *value* | *value* | *value* | *value* |
| XGBoost (tuned)        | *value* | *value* | *value* | *value* | *value* |
