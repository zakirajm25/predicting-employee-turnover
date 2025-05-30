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
7.  [Ethical Considerations](#ethical-considerations)
8.  [Tools and Libraries Used](#tools-and-libraries-used)
9.  [Project Reproducibility](#project-reproducibility)
10. [License](#license)

---

## 1. Project Overview
This project addresses the critical issue of employee attrition at Salifort Motors by applying data science methodologies. Through in-depth exploratory data analysis (EDA) and the development of various predictive machine learning models—including logistic regression, decision trees, and random forests—this initiative aims to uncover the primary drivers of employee turnover. The ultimate goal is to equip Salifort Motors' HR department with a robust model capable of identifying employees at risk of leaving, thereby enabling targeted interventions to improve satisfaction and retention.

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
The analysis is based on the `HR_dataset.csv` file, which contains employee-related data from Salifort Motors. The original dataset comprised 15,000 records and 10 features.

### Dataset Features
The following table details the features available for analysis:

| Feature                 | Description                                                        | Data Type (from notebook) |
| :---------------------- | :----------------------------------------------------------------- | :------------------------ |
| `satisfaction_level`    | Employee-reported job satisfaction level [0–1]                     | `float64`                 |
| `last_evaluation`       | Score of employee's last performance review [0–1]                | `float64`                 |
| `number_project`        | Number of projects employee contributes to                         | `int64`                   |
| `average_montly_hours`  | Average number of hours employee worked per month                  | `int64`                   |
| `time_spend_company`    | How long the employee has been with the company (years)            | `int64`                   |
| `Work_accident`         | Whether or not the employee experienced an accident while at work  | `int64`                   |
| `left`                  | Whether or not the employee left the company (Target Variable)     | `int64`                   |
| `promotion_last_5years` | Whether or not the employee was promoted in the last 5 years     | `int64`                   |
| `Department`            | The employee's department                                          | `object`                  |
| `salary`                | The employee's salary (Categorical: low, medium, high)             | `object`                  |

### Data Cleaning & Preparation
Initial data examination and cleaning involved:
* **Column Renaming:** Standardized column names to `snake_case` (e.g., `Work_accident` to `work_accident`).
* **Missing Value Assessment:** The dataset was confirmed to have no missing values.
* **Duplicate Removal:** Identified and removed 3,008 duplicate entries, resulting in a refined dataset of 11,991 unique records used for analysis.
* **Outlier Identification:** The `tenure` variable exhibited outliers (824 records with tenure outside 1.5-5.5 years). The strategy for addressing these outliers will be determined based on model sensitivity.

### Key EDA Insights & Visualizations
*(Placeholder: This is a critical section. Expand with specific, quantified insights and embed 2-3 key visualizations that tell a story about the data. Examples:)*
* **Attrition Profile:** The cleaned dataset revealed an overall attrition rate of approximately 16.6% (1,991 employees).
    ```
    ![Employees Who Left vs. Stayed](images/left_vs_stayed_ratio.png)
    ```
* **Workload Indicators:** A strong correlation was observed between `average_monthly_hours`, `number_project`, and attrition.
    * *(Placeholder: Specific insight, e.g., "Employees working on 7 projects had a 100% attrition rate.")*
    * *(Placeholder: Specific insight, e.g., "Employees who left were often found in two groups: those working significantly fewer hours (avg. X hrs) or significantly more hours (avg. Y hrs) than their peers.")*
    ```
    [Embed Boxplot/Violin Plot: `average_monthly_hours` by `number_project` segmented by `left` - e.g., `reports/figures/hours_by_projects_attrition.png`]
    ```
* **Satisfaction & Tenure Dynamics:**
    * Employees who left had a mean satisfaction score of 0.44 (median 0.41), compared to 0.67 (median 0.69) for those who stayed.
    * *(Placeholder: Specific insight about tenure, e.g., "A notable dip in satisfaction and spike in attrition occurred around the 4-year tenure mark for employees who left.")*
    ```
    [Embed Boxplot/Violin Plot: `satisfaction_level` by `tenure` segmented by `left` - e.g., `reports/figures/satisfaction_by_tenure_attrition.png`]
    ```
* *(Placeholder: Other significant findings, e.g., from `salary` or `last_evaluation` distributions.)*

---

## 4. Modeling and Evaluation

This section details the process of building and evaluating predictive models for employee attrition.

### Feature Engineering & Preprocessing
*(Placeholder: Describe any feature engineering steps taken, e.g., creation of new features from existing ones. Detail the encoding methods for categorical variables like `department` and `salary` (e.g., One-Hot Encoding, Label Encoding). Specify if and how numerical features were scaled (e.g., StandardScaler, MinMaxScaler). Reiterate the outlier handling strategy for `tenure` if applied at this stage.)*

**Train-Test Split:** The dataset was partitioned into training and testing sets *(Placeholder: specify ratio, e.g., 80% training, 20% testing)* to ensure robust model evaluation on unseen data. A `random_state` was used for reproducibility.

### Model Selection & Rationale
Based on the project's objective to predict a binary outcome (employee leaves or stays), the following classification models were implemented and evaluated:
1.  **Logistic Regression:** Provides a good baseline and interpretable model.
2.  **Decision Tree Classifier:** Useful for understanding decision rules and feature interactions.
3.  **Random Forest Classifier:** An ensemble method known for its robustness and ability to handle non-linear relationships.

### Evaluation Strategy & Metrics
Model performance was primarily assessed using the following metrics, with a focus on correctly identifying employees likely to leave (the positive class, `left=1`):
* **Accuracy:** Overall correctness of predictions.
* **Precision (for class 1):** Of those predicted to leave, how many actually left.
* **Recall (Sensitivity for class 1):** Of those who actually left, how many were correctly identified. (Often a key metric in attrition models to minimize missed at-risk employees).
* **F1-Score (for class 1):** The harmonic mean of Precision and Recall, providing a balanced measure.
* **ROC AUC Score:** Measures the model's ability to distinguish between the two classes.
* **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.

Hyperparameter tuning was performed using `GridSearchCV` to optimize the selected models.

### Model Performance & Champion Model
*(Placeholder:
* Present a clear table summarizing the performance of all tested models (including tuned versions) on the test set, using the metrics above.
* Clearly state which model was chosen as the "champion model" and provide a strong justification based on its performance (especially recall and F1-score for the 'left' class) and interpretability.
* Discuss insights from the champion model, particularly feature importances.)*

[Embed Table: Model Performance Comparison]

[Embed Visualization: ROC Curves for Key Models - e.g., reports/figures/roc_curves.png]

[Embed Visualization: Confusion Matrix for the Champion Model - e.g., reports/figures/champion_confusion_matrix.png]

[Embed Visualization: Feature Importance Plot for the Champion Model - e.g., reports/figures/champion_feature_importance.png]




---

## 5. Key Questions Answered
*(Placeholder: Directly answer the business questions. Example:)*
* **What are the key factors that indicate an employee is likely to leave the company?**
    * *(Answer based on feature importance and EDA, e.g., "The analysis identified [Feature A], [Feature B], and [Feature C] as the strongest predictors of employee attrition.")*
* *(Placeholder: Add other key questions derived from the "Business Understanding" section and answer them with findings.)*

---

## 6. Conclusions & Actionable Recommendations

This project successfully developed a predictive model capable of identifying employees at risk of attrition at Salifort Motors. *(Placeholder: Briefly restate champion model performance, e.g., "The champion [Model Name] achieved an F1-score of A% for predicting leavers.")* The analysis highlights several critical areas for HR intervention:

**Key Conclusions:**
* *(Placeholder: Summarize 2-3 main conclusions from your model and EDA. Example: "Employee workload, as indicated by `average_monthly_hours` and `number_project`, is a significant factor, with overworked employees showing a higher tendency to leave.")*
* *(Placeholder: Conclusion 2, e.g., "Satisfaction levels, particularly around the 4-year tenure mark, are pivotal in retention.")*
* *(Placeholder: Conclusion 3, e.g., "While salary wasn't the primary driver for long-tenured employees, its role in overall job satisfaction and for other segments cannot be ignored.")*

**Actionable Recommendations for Salifort Motors HR:**
1.  **Proactive Workload Management:**
    * Regularly monitor `average_monthly_hours` and `number_project` for all employees.
    * Implement thresholds or alerts for employees exceeding *(Placeholder: e.g., X hours/month or Y projects)* to trigger a review or support discussion.
2.  **Targeted Engagement & Support:**
    * Develop specific engagement programs for employees identified by the model as "at-risk," particularly focusing on those with low `satisfaction_level` or at critical `tenure` points.
    * *(Placeholder: Add a recommendation based on `last_evaluation` if it's a key feature.)*
3.  **Career Development & Recognition:**
    * Review promotion and recognition practices, especially for medium-tenured, high-performing employees who might leave despite high satisfaction.
    * *(Placeholder: Recommendation based on `promotion_last_5years` if significant.)*

**Limitations:**
* The analysis is based on the available quantitative data; qualitative data from exit interviews or employee feedback could provide richer context.
* The dataset had a limited number of employees with very long tenures, potentially affecting model generalization for this specific group.

**Future Work:**
* **Deployment:** Integrate the champion model into HR workflows for real-time risk assessment.
* **Intervention Impact Analysis:** Measure the effectiveness of implemented retention strategies based on model predictions.
* **Data Enrichment:** Explore incorporating additional data points (e.g., manager feedback, commute time, training participation) to refine model accuracy.
* **Continuous Monitoring & Retraining:** Periodically retrain the model with new data to adapt to evolving employment trends.

---

## 7. Ethical Considerations
The application of this predictive model must be approached with a strong ethical framework. Key considerations include:
* **Fairness & Non-Discrimination:** Ensure the model does not perpetuate biases or unfairly target specific demographic groups. Regular audits for bias are recommended.
* **Transparency:** While individual risk scores might be sensitive, the general factors influencing attrition should be communicated to foster a better work environment.
* **Supportive Interventions:** Model predictions should be used to trigger supportive measures (e.g., discussions, workload adjustments, development opportunities) rather than punitive actions.
* **Data Privacy:** Adhere to all data privacy regulations and ensure employee data is handled securely and confidentially.

---

## 8. Tools and Libraries Used
* **Programming Language:** Python 3.x
* **Data Analysis & Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Evaluation:** Scikit-learn, XGBoost
* **Utilities:** math, pickle

---

## 9. Project Reproducibility
To reproduce this analysis:
1.  Clone the repository: `git clone [Your Repository URL]`
2.  Navigate to the project directory: `cd analysis-and-predicting-employee-turnover-various-models`
3.  Set up the Python environment using the provided `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    *(Alternatively, if an `environment.yml` is provided: `conda env create -f environment.yml`)*
4.  Ensure the `HR_capstone_dataset.csv` file is located in the `data/raw/` directory.
5.  Execute the Jupyter Notebook `notebooks/Activity__Course_7_Salifort_Motors_project_lab.ipynb`.

---

## 10. License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
