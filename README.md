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
   
    ![Employees Who Left vs. Stayed](images/1_left_vs_stayed_ratio.png)

* **Workload Indicators:** A strong correlation was observed between `average_monthly_hours`, `number_project`, and attrition.
    * *(Placeholder: Specific insight, e.g., "Employees working on 7 projects had a 100% attrition rate, while those with 3-4 projects showed higher retention.")*
    * *(Placeholder: Specific insight, e.g., "Employees who left were often found in two groups: those working significantly fewer hours (avg. X hrs) or significantly more hours (avg. Y hrs) than their peers.")*
    
    ![Monthly Hours by Number of Projects, Segmented by Attrition](images/2_monthly_hours_by_projects.png)
  
* **Satisfaction & Tenure Dynamics:**
    * Employees who left had a mean satisfaction score of 0.44 (median 0.41), compared to 0.67 (median 0.69) for those who stayed.
    * *(Placeholder: Specific insight about tenure, e.g., "A notable dip in satisfaction and spike in attrition occurred around the 4-year tenure mark for employees who left.")*
      
    ![Satisfaction Level by Tenure, Segmented by Attrition](images/3_satisfaction_by_tenure.png)
  
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
[Embed Visualization: ROC Curves for Key Models - e.g., `images/roc_curves.png`]
[Embed Visualization: Confusion Matrix for the Champion Model - e.g., `images/champion_confusion_matrix.png`]
[Embed Visualization: Feature Importance Plot for the Champion Model - e.g., `images/champion_feature_importance.png`]
```
---

## 5. Key Questions Answered
*(Placeholder: Directly answer the business questions based on your final model and EDA. Example:)*
* **What are the key factors that indicate an employee is likely to leave Salifort Motors?**
    * *(Answer based on feature importance from your champion model and EDA insights, e.g., "The analysis identified `satisfaction_level`, `average_monthly_hours`, `number_project`, and `tenure` as the strongest predictors of employee attrition. Specifically, [add a brief elaboration].")*
* *(Placeholder: What is the profile of an employee likely to leave?)*
    * *(Answer based on characteristics highlighted by the model and EDA.)*

---

## 6. Conclusions & Actionable Recommendations

### Key Conclusions
This project successfully developed and evaluated predictive models for employee attrition at Salifort Motors. *(Placeholder: The champion [Model Name, e.g., Random Forest/XGBoost model] demonstrated strong performance, achieving an F1-score of A% and a Recall of B% for predicting employees who would leave.)* The analysis reveals several critical areas impacting employee retention:
* *(Placeholder: Summarize 2-3 main conclusions from your champion model and EDA. Example: "Low employee satisfaction, particularly around the four-year tenure mark, is a major precursor to attrition.")*
* *(Placeholder: Conclusion 2, e.g., "Extreme workloads, characterized by very high average monthly hours and a high number of assigned projects (especially 6-7 projects), significantly increase the likelihood of an employee leaving.")*
* *(Placeholder: Conclusion 3, e.g., "Performance evaluation scores (`last_evaluation`) also play a role in predicting turnover, suggesting a link between perceived performance/recognition and an employee's decision to stay or leave.")*

### Actionable Recommendations for Salifort Motors HR
Based on these findings, the following actions are recommended:
1.  **Implement Proactive Workload Management Systems:**
    * Regularly monitor `average_monthly_hours` and `number_project`. Aim for an optimal range of 3-4 projects and 150-250 average monthly hours per employee.
    * Develop an alert system or review process for employees consistently exceeding these thresholds to discuss workload and provide support.
2.  **Targeted Engagement for At-Risk Segments:**
    * Focus retention efforts on employees identified by the model as high-risk, particularly those with low `satisfaction_level` and those approaching or at the four-year `tenure` mark.
    * Conduct "stay interviews" or targeted surveys for these segments to understand concerns and address them proactively.
3.  **Review and Enhance Performance Evaluation & Recognition Processes:**
    * Ensure `last_evaluation` scores are fair, transparent, and used constructively. Provide clear feedback and development paths based on evaluations.
    * *(Placeholder: If `promotion_last_5years` is significant, add a recommendation: "Review promotion criteria and frequency to ensure high-performing employees are recognized and see a growth path within the company.")*
    * Consider a proportionate scale for rewarding employees who consistently contribute more or manage higher workloads, especially those working over 200 hours per month.

### Limitations
* The analysis is based on the available quantitative data; incorporating qualitative data (e.g., exit interview feedback, employee survey comments) could provide richer context to the drivers of attrition.
* The dataset's specific timeframe is not explicitly defined, which might limit the generalization of findings to different economic or company periods.
* Outliers in `tenure` were retained; while tree-based models are robust, these could slightly influence specific interpretations for very long or very short-tenured employees.

### Future Work
* **Model Deployment & Integration:** Develop a user-friendly interface or integrate the champion model into existing HR systems for ongoing, real-time risk assessment.
* **Intervention Impact Analysis:** Design and implement pilot programs based on the recommendations, then measure their impact on attrition rates to validate and refine strategies.
* **Predicting Other HR Metrics:** Explore building models to predict related outcomes like employee performance scores or satisfaction levels, using these as leading indicators.
* **Unsupervised Learning for Deeper Segmentation:** Apply clustering techniques (e.g., K-means) to identify distinct employee archetypes or segments, which could inform more nuanced HR interventions.
* **Continuous Monitoring & Model Retraining:** Periodically retrain the model with new employee data to maintain its predictive accuracy and adapt to evolving workforce dynamics.

---

## 7. Ethical Considerations
The application of predictive analytics in HR must be handled with utmost care and ethical consideration:
* **Fairness & Bias Mitigation:** The model should be regularly audited to ensure it does not perpetuate or introduce biases against any protected groups. Predictions should not be the sole basis for decisions.
* **Transparency & Communication:** While individual risk scores are sensitive, the general factors identified as contributing to attrition can be used to transparently communicate areas of focus for improving the overall work environment.
* **Purpose Limitation & Supportive Use:** Model insights should be used to trigger supportive interventions (e.g., discussions about workload, development opportunities, wellness programs) rather than for adverse or punitive actions against employees.
* **Data Privacy & Security:** All employee data must be handled in compliance with privacy regulations (e.g., GDPR, CCPA), ensuring confidentiality and security throughout the data lifecycle.

---

## 8. Tools and Libraries Used
* **Programming Language:** Python 3.x
* **Data Analysis & Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Evaluation:** Scikit-learn (for Logistic Regression, Decision Trees, Random Forests, GridSearchCV, train_test_split, and various metrics), XGBoost
* **Utilities:** math (built-in), pickle (built-in for model saving)

---

## 9. Project Reproducibility
To reproduce this analysis:
1.  Clone the repository: `git clone [Your Repository URL Here]`
2.  Navigate to the project directory: `cd [Your Repository Name Here]`
3.  Set up the Python environment. It is recommended to use a virtual environment (e.g., Conda or venv). Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  Ensure the dataset `HR_capstone_dataset.csv` is placed in the `data/raw/` directory within the project.
5.  Open and run the Jupyter Notebook `notebooks/Activity__Course_7_Salifort_Motors_project_lab.ipynb` (or your specific notebook name).

---

## 10. License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
