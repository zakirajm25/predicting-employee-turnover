# Analysis and Predicting Employee Turnover Using Various Models

**Short Description:** Analyzing employee data to predict turnover using various machine learning models. Focus on HR insights for Salifort Motors.

---

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Business Understanding](#business-understanding)
3.  [Data Understanding](#data-understanding)
    * [Data Source](#data-source)
    * [Dataset Features](#dataset-features)
    * [Initial Data Cleaning & Preparation](#initial-data-cleaning--preparation)
    * [Exploratory Data Analysis (EDA) Highlights](#exploratory-data-analysis-eda-highlights)
4.  [Modeling and Evaluation](#modeling-and-evaluation)
    * [Data Preprocessing for Modeling](#data-preprocessing-for-modeling)
    * [Models Implemented](#models-implemented)
    * [Evaluation Metrics](#evaluation-metrics)
    * [Model Performance & Champion Model](#model-performance--champion-model)
5.  [Conclusion and Recommendations](#conclusion-and-recommendations)
6.  [Ethical Considerations](#ethical-considerations)
7.  [Tools and Libraries](#tools-and-libraries)
8.  [Getting Started / Reproducibility](#getting-started--reproducibility)
9.  [License](#license)

---

## Project Overview
This project analyzes employee data from Salifort Motors to identify key factors contributing to employee attrition. By building predictive models, the aim is to provide actionable, data-driven insights to the Human Resources (HR) department. The primary goal is to predict whether an employee will leave the company, thereby enabling proactive measures to improve employee satisfaction and retention. The analysis utilizes regression and tree-based machine learning models, with a champion model selected based on various performance metrics.

---

## Business Understanding
**Stakeholder:** HR Department, Salifort Motors

**Problem:** The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They are seeking data-driven suggestions based on an understanding of the data. The central business question is: "What’s likely to make an employee leave the company?"

**Goal:**
* Analyze the data collected by the HR department.
* Build a model that predicts whether or not an employee will leave the company.
* Provide data-driven suggestions to improve employee retention.

Predicting employees likely to quit might make it possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

---

## Data Understanding

### Data Source
The dataset used is `HR_capstone_dataset.csv`, containing employee information from Salifort Motors.

### Dataset Features
The dataset initially contains 15,000 rows and 10 columns:

| Variable                | Description                                                        | Data Type (from notebook) |
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

### Initial Data Cleaning & Preparation
* **Column Renaming:** Column names were standardized to `snake_case` for better consistency (e.g., "Work_accident" to `work_accident`, "average_montly_hours" to `average_monthly_hours`, "time_spend_company" to `tenure`, "Department" to `department`).
* **Missing Values:** The notebook output indicates no missing values (`df0.isnull().sum()` shows 0 for all columns).
* **Duplicate Removal:** The notebook identified 3,008 duplicate rows, which were subsequently dropped, resulting in a cleaned dataset (`df1`) for further analysis.
* **Outlier Detection:** Outliers were noted in the `tenure` variable. The notebook shows a calculation for upper (5.5 years) and lower (1.5 years) limits, identifying 824 rows with outliers in `tenure`. The handling of these outliers will be considered during modeling.

### Exploratory Data Analysis (EDA) Highlights
*(Placeholder: This section should be expanded with key findings and visualizations from your EDA once completed. Based on the notebook's plan, this would include:)*
* Distribution of employees who left vs. stayed (Overall attrition rate: 1991 employees left, 10000 stayed, meaning ~16.6% attrition in the cleaned dataset).
* Relationship between `average_monthly_hours` and `number_project`, segmented by `left` status.
    * *(Placeholder: Insights from the boxplot and histogram on this relationship, e.g., overworked employees leaving, optimal project numbers.)*
* Relationship between `satisfaction_level` and `tenure`, segmented by `left` status.
    * *(Placeholder: Insights on satisfaction levels for different tenure groups and their attrition.)*
* Mean and median satisfaction scores for employees who left vs. stayed.
    * Stayed: Mean 0.667, Median 0.69
    * Left: Mean 0.440, Median 0.41
* Salary levels for different tenures.
    * *(Placeholder: Insights from the salary histogram by tenure.)*
* Correlation between `average_monthly_hours` and `last_evaluation`.
    * *(Placeholder: Insights from the scatterplot.)*
