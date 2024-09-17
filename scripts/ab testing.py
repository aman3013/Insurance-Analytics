import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def convert_to_numeric(df, columns):
    """Convert specified columns to numeric, forcing errors to NaN."""
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def handle_missing_values(df, columns, strategy='mean'):
    """Handle missing values in specified columns."""
    if strategy == 'mean':
        for column in columns:
            df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == 'zero':
        for column in columns:
            df[column].fillna(0, inplace=True)
    return df

def identify_outliers(df, column):
    """Identify outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def cap_outliers(df, column):
    """Cap outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def analyze_outliers(df, columns):
    """Analyze and cap outliers for specified columns."""
    for column in columns:
        outliers = identify_outliers(df, column)
        print(f"\nOutliers in {column}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Percentage of outliers: {(len(outliers) / len(df)) * 100:.2f}%")
        if len(outliers) > 0:
            print(outliers[column].describe())
        print("\n" + "-"*50)
        df = cap_outliers(df, column)
    return df

def perform_ttest(df, groupA_col, groupB_col):
    """Perform t-test between two groups."""
    groupA = df[groupA_col]
    groupB = df[groupB_col]
    t_stat, p_value = ttest_ind(groupA, groupB, nan_policy='omit')
    return t_stat, p_value

def perform_chi2_test(df, row_col, col_col):
    """Perform Chi-Square test on contingency table."""
    contingency_table = pd.crosstab(df[row_col], df[col_col])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return chi2_stat, p_value, dof, expected

def analyze_p_value(p_value, hypothesis_name):
    """Analyze p-value to determine hypothesis rejection."""
    if p_value < 0.05:
        print(f"Reject the null hypothesis for {hypothesis_name}.")
    else:
        print(f"Fail to reject the null hypothesis for {hypothesis_name}.")

def main():
    # Load data
    file_path = '../data/MachineLearningRating_v3_cleaned.csv'
    df = load_data(file_path)
    
    # Convert columns to numeric and handle missing values
    numeric_columns = ['TotalPremium', 'TotalClaims']
    df = convert_to_numeric(df, numeric_columns)
    df = handle_missing_values(df, numeric_columns, strategy='zero')
    
    # Analyze and cap outliers
    columns_to_clean = ['cubiccapacity', 'kilowatts', 'CapitalOutstanding', 'SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium', 'TotalClaims']
    df = analyze_outliers(df, columns_to_clean)
    
    # Calculate risk
    df['Risk'] = df['TotalPremium'] - df['TotalClaims']
    
    # Perform t-tests
    t_stat_province, p_value_province = perform_ttest(df[df['Province'] == 'Gauteng']['Risk'], df[df['Province'] == 'Western Cape']['Risk'])
    t_stat_zip, p_value_zip = perform_ttest(df[df['PostalCode'] == 7560]['Risk'], df[df['PostalCode'] == 4067]['Risk'])
    t_stat_margin, p_value_margin = perform_ttest(df[df['PostalCode'] == 7560]['TotalPremium'] - df[df['PostalCode'] == 7560]['TotalClaims'], df[df['PostalCode'] == 4067]['TotalPremium'] - df[df['PostalCode'] == 4067]['TotalClaims'])
    t_stat_gender, p_value_gender = perform_ttest(df[df['Gender'] == 'Female']['Risk'], df[df['Gender'] == 'Male']['Risk'])
    
    # Analyze t-test p-values
    analyze_p_value(p_value_province, "Risk Differences Across Provinces")
    analyze_p_value(p_value_zip, "Risk Differences Between Zip Codes")
    analyze_p_value(p_value_margin, "Margin Differences Between Zip Codes")
    analyze_p_value(p_value_gender, "Risk Differences Between Women and Men")
    
    # Perform Chi-Square tests
    chi2_stat_provinces, p_value_provinces, dof_provinces, expected_provinces = perform_chi2_test(df, 'Province', 'ClaimMade')
    chi2_stat_zip, p_value_zip, dof_zip, expected_zip = perform_chi2_test(df[df['PostalCode'].isin(df['PostalCode'].value_counts().nlargest(5).index)], 'PostalCode', 'ClaimMade')
    chi2_stat_gender, p_value_gender, dof_gender, expected_gender = perform_chi2_test(df, 'Gender', 'ClaimMade')
    
    # Analyze Chi-Square p-values
    analyze_p_value(p_value_provinces, "Risk Differences Across Provinces")
    analyze_p_value(p_value_zip, "Risk Differences Between Zip Codes")
    analyze_p_value(p_value_gender, "Risk Differences Between Women and Men")

if __name__ == "__main__":
    main()
