import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


# Load data
df = pd.read_csv('employee_data.csv')

# 1.Setting Goals and Identifying Data Needs
#Make sure the available data can answer the business questions.


# 2. Identifying Data Structure
print(df.head())
print(df.dtypes)
print(df.shape)

# 3. Descriptive Statistics Analysis
print(df.describe())

# Frequency analysis 'Gender' Column
gender_freq = df['Gender'].value_counts()

# Frequency analysis 'Department' Column
department_freq = df['Department'].value_counts()

# Frequency analysis 'Position' Column
position_freq = df['Position'].value_counts()

# Frequency analysis 'Education' Column
education_freq = df['Education'].value_counts()


print("Frequency Gender:")
print(gender_freq)

print("\nFrequency Department:")
print(department_freq)

print("\nFrequency Position:")
print(position_freq)

print("\nFrequency Education:")
print(education_freq)
  
print(df.info())


# 4. Identifying Missing Values
print(df.isnull().sum())

# 5. Identifying Duplicates
print(df.duplicated().sum())
#df = df.drop_duplicates()

# 6. Identifying Distribution Data, Outliers and Correlation
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import numpy as np


# select the appropriate numerical columns
numerical_cols = ['Age', 'Satisfaction_Score', 'Tenure_Years', 'Salary', 'Hours_per_Week', 'Performance_Score']

print("Identifying Outliers with Z-Scores")

# Calculating Z-Score on numerical columns
z_scores = np.abs(zscore(df[numerical_cols]))

# Identifikasi outliers based on Z-Score > 3
outliers_z = (z_scores > 3)

# Display columns with outliers
outliers_summary_z = outliers_z.any(axis=0).values  # convert into numpy array boolean
outliers_columns_z = np.array(numerical_cols)[outliers_summary_z]  # using numpy array for indexing

print("Columns that have outliers based on Z-Score:")
print(outliers_columns_z)

# Identifying columns that have outliers.
for col in outliers_columns_z:
    print(f"{col} have {outliers_z[:, numerical_cols.get_loc(col)].sum()} outliers based on Z-Score")
   
   
#or...

print("Identifying Outliers with IQR")

# Calculation Q1 (Quartil 1) dan Q3 (Quartil 3)
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

# Setting lower and upper bound
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identifying columns that have outliers.
outliers = ((df[numerical_cols] < lower_bound) | (df[numerical_cols] > upper_bound))

# Display columns that have outliers.
outliers_summary = outliers.any()
outliers_columns = outliers_summary[outliers_summary].index
print("Columns that have outliers based on IQR:")
print(outliers_columns)

# Display outliers per kolom
for col in outliers_columns:
    print(f"{col} have {outliers[col].sum()} outliers")

#or...

print("Identifying Outliers with Visualization")
for col in numerical_cols:
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(df[col])
    plt.xticks(rotation=45)
    plt.title('Boxplot untuk mendeteksi Outliers')
    plt.show()


# Visualization Data Analysis Univariate and Bivariate
# Plot a histogram and KDE to show the frequency distribution and observe the pattern of the distribution.

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

    # Skewness dan Kurtosis
    skewness = skew(df[col].dropna())
    kurt = kurtosis(df[col].dropna())
    print(f'{col} - Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f}')

# Plot distribution
categorical_cols = ['Gender', 'Department', 'Position', 'Turnover']

for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


# Analisis Bivariate (2 variables)
# Boxplot: Age vs Turnover
plt.figure(figsize=(8, 5))
sns.boxplot(x='Turnover', y='Age', data=df)
plt.title('Distribution Age based on Turnover (Resign vs Stay)')
plt.xticks([0,1],['Stay','Resign'])
plt.show()


# Boxplot: Salary vs Turnover
plt.figure(figsize=(8, 5))
sns.boxplot(x='Turnover', y='Salary', data=df)
plt.title('Distribution Salary based on Turnover (Resign vs Stay)')
plt.xticks([0,1],['Stay','Resign'])
plt.show()

# Boxplot: Satisfaction_Score vs Turnover
plt.figure(figsize=(8, 5))
sns.boxplot(x='Turnover', y='Satisfaction_Score', data=df)
plt.title('Distribution Satisfaction Score based on Turnover (Resign vs Stay)')
plt.xticks([0,1],['Stay','Resign'])
plt.show()

# Boxplot: Hours_per_Week vs Turnover
plt.figure(figsize=(8, 5))
sns.boxplot(x='Turnover', y='Hours_per_Week', data=df)
plt.title('Distribution Hours per week based on Turnover (Resign vs Stay)')
plt.xticks([0,1],['Stay','Resign'])
plt.show()

# Boxplot: Tenure_Years vs Turnover
plt.figure(figsize=(8, 5))
sns.boxplot(x='Turnover', y='Tenure_Years', data=df)
plt.title('Distribution Tenure Years based on Turnover (Resign vs Stay)')
plt.xticks([0,1],['Stay','Resign'])
plt.show()


#Barplot: Department vs Turnover
plt.figure(figsize=(8, 5))
sns.barplot(x='Department', y='Turnover', data=df, estimator=sum)
plt.title('Turnover by Department')
plt.xticks(rotation=45)
plt.show()

#Barplot: Position vs Turnover
plt.figure(figsize=(8, 5))
sns.barplot(x='Position', y='Turnover', data=df, estimator=sum)
plt.title('Turnover by Position')
plt.xticks(rotation=45)
plt.show()

#Barplot: Gender vs Turnover
plt.figure(figsize=(8, 5))
sns.barplot(x='Gender', y='Turnover', data=df, estimator=sum)
plt.title('Turnover by Gender')
plt.xticks(rotation=45)
plt.show()


# Scatter Plot: Age vs Salary   
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Salary', hue='Turnover', data=df)
plt.title('Age vs Salary with Turnover')
plt.show()

#............


#Visualisasi correlation

# Select columns
data = df[['Salary', 'Tenure_Years', 'Hours_per_Week', 'Satisfaction_Score']]

# Calculating Matrix Correlation
correlation_matrix = data.corr(method='pearson')

# Display Matrix Correlation
print(correlation_matrix)

# Heatmap Correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# 7. Identifying Value Mapping
# Mapping Categorical columns, Standaritation and mormalization columns



# 8. Create a Data Dictionary
# Create documentation columns 


# 9. Create a Transformation Plan
# Create a transformation plan based on discovery process
