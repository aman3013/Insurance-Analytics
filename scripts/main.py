import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_analysis():
    # Load the dataset
    df = pd.read_csv('../data/MachineLearningRating_v3.csv')

    # Drop unnecessary columns
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Drop columns with high missing percentage
    df.drop(['CustomValueEstimate', 'CrossBorder', 'NumberOfVehiclesInFleet', 'WrittenOff', 'Rebuilt', 'Converted'], axis=1, inplace=True)

    # Impute missing values
    df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')
    df['CapitalOutstanding'] = df['CapitalOutstanding'].fillna(df['CapitalOutstanding'].mean())
    df['Bank'] = df['Bank'].fillna(df['Bank'].mode()[0])
    df['AccountType'] = df['AccountType'].fillna(df['AccountType'].mode()[0])
    df['MaritalStatus'] = df['MaritalStatus'].fillna(df['MaritalStatus'].mode()[0])
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

    # Forward fill missing values
    df['mmcode'] = df['mmcode'].ffill()
    df['VehicleType'] = df['VehicleType'].ffill()
    df['make'] = df['make'].ffill()
    df['Model'] = df['Model'].ffill()

    # Handle vehicle-related columns
    vehicle_cols = ['Cylinders', 'cubiccapacity', 'kilowatts', 'bodytype', 'NumberOfDoors', 'VehicleIntroDate']
    for col in vehicle_cols:
        df[col] = df[col].ffill().bfill()  # Forward fill, then backward fill
        if df[col].isnull().sum() > 0:  # If there are still missing values
            df[col] = df[col].fillna(df[col].mode()[0])  # Use mode imputation

    # Handle NewVehicle column as categorical
    df['NewVehicle'] = df['NewVehicle'].fillna('Unknown')

    # Check for any remaining missing values
    print(df.isnull().sum())

    # Optionally save the cleaned dataset
    df.to_csv('../data/MachineLearningRating_v3_cleaned.csv', index=False)

    # Data Summarization
    print("Data Summarization:")
    print(df.head())
    print(df.info())
    print(df.describe())

    # Descriptive Statistics
    numerical_features = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 'CapitalOutstanding']
    print("Descriptive Statistics for Numerical Features:")
    print(df[numerical_features].describe())

    # Function to safely convert to numeric
    def safe_convert_to_numeric(series):
        return pd.to_numeric(series, errors='coerce')

    # Convert to numeric and check which conversions were successful
    converted_features = {}
    for feature in numerical_features:
        converted = safe_convert_to_numeric(df[feature])
        if not converted.isna().all():  # If not all values are NaN after conversion
            converted_features[feature] = converted
        else:
            print(f"Warning: Could not convert '{feature}' to numeric. Skipping this feature.")

    # Calculate variability measures for successfully converted features
    print("\nVariability Measures:")
    for feature, series in converted_features.items():
        print(f"\n{feature}:")
        print(f"Range: {series.max() - series.min()}")
        print(f"Interquartile Range (IQR): {series.quantile(0.75) - series.quantile(0.25)}")
        print(f"Variance: {series.var()}")
        print(f"Standard Deviation: {series.std()}")
        print(f"Coefficient of Variation: {series.std() / series.mean()}")

    # Print data types of original columns
    print("\nOriginal Data Types:")
    for feature in numerical_features:
        print(f"{feature}: {df[feature].dtype}")

    # Print unique values for columns that couldn't be converted (if any)
    for feature in set(numerical_features) - set(converted_features.keys()):
        print(f"\nUnique values in {feature}:")
        print(df[feature].unique())

    # Data Structure
    print("\nData Structure:")
    print(df.dtypes)
    display(df[numerical_features])

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

   # Create histograms for numerical columns
    for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.show()

        # Create count plots for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
            top_categories = df[col].value_counts().nlargest(10).index  # Adjust the number as needed
            plt.figure(figsize=(12, 6))
            sns.countplot(y=df[col].loc[df[col].isin(top_categories)], order=top_categories)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.show()

        # Create scatter plot of TotalPremium vs TotalClaims
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TotalPremium', y='TotalClaims', alpha=0.7)
    plt.title('Scatter Plot of TotalPremium vs TotalClaims')
    plt.xlabel('TotalPremium')
    plt.ylabel('TotalClaims')
    plt.show()

        # Compute correlation matrix
    correlation_matrix = df[['TotalPremium', 'TotalClaims']].corr()

        # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Correlation Matrix for TotalPremium and TotalClaims')
    plt.show()

        # Create box plots for TotalPremium and TotalClaims by categorical columns
    for col in categorical_cols:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x=col, y='TotalPremium')
            plt.title(f'Boxplot of TotalPremium by {col}')
            plt.xlabel(col)
            plt.ylabel('TotalPremium')
            plt.xticks(rotation=45)
            plt.show()

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x=col, y='TotalClaims')
            plt.title(f'Boxplot of TotalClaims by {col}')
            plt.xlabel(col)
            plt.ylabel('TotalClaims')
            plt.xticks(rotation=45)
            plt.show()

        # Identify numerical columns
    numerical_cols = ['TransactionMonth', 'CapitalOutstanding', 'SumInsured', 
                          'CalculatedPremiumPerTerm', 'ExcessSelected', 'TotalPremium', 
                          'TotalClaims', 'RegistrationYear', 'Cylinders', 'cubiccapacity', 
                          'kilowatts', 'NumberOfDoors']

        # Convert columns to numeric values
    for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values
    df = df.dropna(subset=numerical_cols)

        # Create box plots for numerical columns
    fig, axs = plt.subplots(1, len(numerical_cols), figsize=(20, 6))

    for i, col in enumerate(numerical_cols):
            axs[i].boxplot(df[col])
            axs[i].set_title(col)

    plt.show()

        # Calculate outlier thresholds for each numerical column
    outlier_thresholds = {}

    for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_threshold = Q1 - 1.5 * IQR
            upper_threshold = Q3 + 1.5 * IQR
            outlier_thresholds[col] = (lower_threshold, upper_threshold)

    print(outlier_thresholds)

        # Identify outliers for each numerical column
    outliers = {}

    for col, thresholds in outlier_thresholds.items():
            lower_threshold, upper_threshold = thresholds
            outliers[col] = df[(df[col] < lower_threshold) | (df[col] > upper_threshold)]

    print(outliers)

        # Visualize outliers for each numerical column
    fig, axs = plt.subplots(1, len(numerical_cols), figsize=(20, 6))

    for i, col in enumerate(numerical_cols):
            axs[i].boxplot(df[col])
            axs[i].scatter(outliers[col].index, outliers[col][col], color='red')
            axs[i].set_title(col)

    plt.show()
        # Create a pair plot to visualize relationships between numerical features
    sns.pairplot(df[numerical_cols], diag_kind='kde', markers='o', palette='husl')
    plt.suptitle('Pair Plot of Numerical Features', y=1.02)
    plt.show()

        # Calculate the correlation matrix
    correlation_matrix = df[numerical_cols].corr()

        # Create a heatmap to visualize correlations
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap of Numerical Features')
    plt.show()

        # Create a violin plot to compare the distribution of TotalPremium across different RegistrationYears
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='RegistrationYear', y='TotalPremium', data=df, palette='muted')
    plt.title('Distribution of Total Premium by Registration Year')
    plt.xlabel('Registration Year')
    plt.ylabel('Total Premium')
    plt.xticks(rotation=45)
    plt.show()

data_analysis()