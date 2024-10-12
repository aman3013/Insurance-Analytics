# Insurance-Analytics

## Project Overview

AlphaCare Insurance Solutions (ACIS) is committed to advancing risk and predictive analytics to optimize car insurance planning and marketing in South Africa. As a new member of the data analytics team, your role is to analyze historical insurance claim data, with the objective of identifying “low-risk” targets and optimizing marketing strategies. By identifying potential clients who qualify for reduced premiums, ACIS can attract new customers and enhance their market position.

This project will help you apply and strengthen skills in data engineering, predictive analytics, statistical modeling, and machine learning.


## Business Objective

The main goals of this project are:
1. **Optimize Marketing Strategies**: Use historical data to help improve targeting and strategy decisions.
2. **Discover Low-Risk Clients**: Identify clients who may benefit from reduced premiums, presenting a marketing opportunity.
3. **Risk Analytics**: Analyze and model insurance risk factors to inform pricing strategies.


## Learning Outcomes

Throughout this project, you will:
- Understand and extract insights from complex insurance data.
- Develop statistical models to predict insurance risk and premium rates.
- Apply A/B hypothesis testing and machine learning algorithms to solve real-world business problems.
- Enhance skills in Python programming, data version control (DVC), model versioning, and MLOps.
- Utilize feature importance techniques such as SHAP and LIME to interpret machine learning models.


## Tasks Breakdown

### Task 1: Git and GitHub Setup

- **Objectives**: Set up version control and conduct Exploratory Data Analysis (EDA).
- **Steps**:
  - Create a GitHub repository and initialize Git version control.
  - Set up a CI/CD pipeline with GitHub Actions.
  - Perform EDA to understand the data structure and quality, including:
    - Descriptive statistics for numerical features.
    - Univariate and multivariate analysis.
    - Outlier detection and data visualization.


### Task 2: Data Version Control (DVC)

- **Objectives**: Set up data version control for efficient data tracking.
- **Steps**:
  - Install and initialize DVC in your project directory.
  - Configure local storage for dataset versioning.
  - Track data with DVC, push data to remote storage, and commit versioned changes to GitHub.


### Task 3: A/B Hypothesis Testing

- **Objectives**: Test for significant differences in insurance risks across various client segments.
- **Steps**:
  - Define key performance indicators (KPIs) for hypothesis testing.
  - Segment data into control and test groups (Group A and Group B).
  - Perform statistical tests (e.g., chi-squared tests, t-tests) to determine the impact of various factors (e.g., location, gender) on insurance risks.
  - Analyze p-values to accept or reject null hypotheses.
  - Document findings and present insights on business strategy.


### Task 4: Statistical Modeling

- **Objectives**: Build and evaluate predictive models for insurance risk and premium predictions.
- **Steps**:
  - Handle missing data and apply feature engineering.
  - Encode categorical data to make it suitable for modeling.
  - Implement multiple models, including:
    - Linear Regression
    - Random Forests
    - XGBoost
  - Evaluate models based on accuracy, precision, recall, and F1-score.
  - Use SHAP or LIME to explain model predictions and analyze feature importance.


## Data Overview

The dataset spans from February 2014 to August 2015, including information about:
- **Client Data**: Citizenship, marital status, gender, location, etc.
- **Car Data**: Make, model, year, kilowatts, etc.
- **Insurance Policy**: Premiums, claims, coverage type, etc.
- **Claims**: TotalPremium, TotalClaims, etc.

Key columns:
- `PolicyID`, `TransactionMonth`, `TotalPremium`, `TotalClaims`, `Province`, `Gender`, `ZipCode`


## Key Performance Indicators (KPIs)

- **Git Version Control**: Proper use of branches, commit frequency, and CI/CD pipeline integration.
- **EDA Proactivity**: The quality of exploratory analysis, including handling missing data, detecting outliers, and visualizing key insights.
- **Hypothesis Testing**: Clear understanding and application of statistical testing methods with actionable results.
- **Modeling**: Accurate and interpretable machine learning models, with documented feature importance.


## Competency Mapping

The tasks in this challenge will contribute to essential job competencies in:
- **Data Engineering**: Data transformation, versioning, and EDA techniques.
- **Predictive Analytics**: Statistical testing, feature engineering, and modeling.
- **Machine Learning Engineering**: Model building, evaluation, and interpretability.


## Project Deliverables

1. **GitHub Repository**: Hosting all code, data, and versioning files.
2. **EDA Report**: Visual insights and key findings.
3. **DVC Setup**: Proper tracking and versioning of datasets.
4. **A/B Hypothesis Testing Results**: Documented statistical tests and business recommendations.
5. **Modeling Report**: Comparison of different models, evaluation metrics, and feature importance analysis.

