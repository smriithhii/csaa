
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Read your dataset
odf = pd.read_csv('/content/Obfuscated-MalMem2022.csv')
num_features = odf.shape[1]
print("Number of features:", num_features)

# Drop columns with all entries equal to 0
zero_cols = [col for col in odf.columns if (odf[col] == 0).all()]
odf.drop(columns=zero_cols, inplace=True)

# Print the updated dataset
print(zero_cols)

# Print the number of dropped columns
print("Number of dropped columns:", len(zero_cols))

# Check for missing values
missing_values = odf.isnull().sum()
print("\nMissing values per column:")
print(missing_values.head())
print("We won't be eliminating any columns as none of them have missing values (checked with all the values, so now I am printing only 5 of them)")

# Check for duplicate rows
duplicate_rows = odf[odf.duplicated()]
print("\nNumber of duplicate rows:", len(duplicate_rows))
odf.drop_duplicates(inplace=True)
print("\nDuplicate rows eliminated")

# Check the data types of each column
numeric_cols = odf.select_dtypes(include=['number']).columns

# Standardize numeric columns
scaler = StandardScaler()
odf[numeric_cols] = scaler.fit_transform(odf[numeric_cols])

# Normalize numeric columns
min_max_scaler = MinMaxScaler()
odf[numeric_cols] = min_max_scaler.fit_transform(odf[numeric_cols])

# Print the standardized and normalized dataset
print("\nStandardized and normalized dataset is created")

# Recursive Feature Elimination (RFE)
X = odf.drop('Class', axis=1)
y = odf['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop non-numeric columns before normalization
non_numeric_cols = X.select_dtypes(exclude='number').columns
X_train_numeric = X_train.drop(columns=non_numeric_cols)
X_test_numeric = X_test.drop(columns=non_numeric_cols)

# Initialize a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize RFE
rfe = RFE(model, n_features_to_select=10)  # Choose the number of features you want to select

# Fit RFE
fit = rfe.fit(X_train_numeric, y_train)

# Get selected features
selected_features_rfe = X_train_numeric.columns[fit.support_]

# Train a model with the selected features
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
model_rfe.fit(X_train[selected_features_rfe], y_train)

# Make predictions on the test set
y_pred_rfe = model_rfe.predict(X_test[selected_features_rfe])

# Evaluate the model
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print("Accuracy with RFE selected features:", accuracy_rfe)

# SelectKBest with chi-squared test
X = odf.drop('Class', axis=1)
y = odf['Class']

# Convert any categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SelectKBest with chi-squared test
k_best = SelectKBest(chi2, k=10)  # Choose the number of features you want to select

# Fit SelectKBest
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Get selected features indices
selected_feature_indices = k_best.get_support(indices=True)

# Train a model with the selected features
model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
model_selected.fit(X_train_kbest, y_train)

# Make predictions on the test set
y_pred_selected = model_selected.predict(X_test_kbest)

# Evaluate the model
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with SelectKBest selected features:", accuracy_selected)

# The second part of your code
odf = pd.read_csv('C:\\Users\\smrit\\OneDrive\\Desktop\\sem6\\CSA\\Obfuscated-MalMem2022.csv')
num_features = odf.shape[1]
print("Number of features:", num_features)

# Drop columns with all entries equal to 0
zero_cols = [col for col in odf.columns if (odf[col] == 0).all()]
odf.drop(columns=zero_cols, inplace=True)

# Print the updated dataset
print(zero_cols)

# Print the number of dropped columns
print("Number of dropped columns:", len(zero_cols))

# Check for missing values
missing_values = odf.isnull().sum()
print("\nMissing values per column:")
print(missing_values.head())
print("We wont be eliminating any columns as none of them have missing values (checked with all the values so now i am printing only 5 of them)")

# Check for duplicate rows
duplicate_rows = odf[odf.duplicated()]
print("\nNumber of duplicate rows:", len(duplicate_rows))
odf.drop_duplicates(inplace=True)
print("\nDuplicate rows eliminated")

# Check the data types of each column
numeric_cols = odf.select_dtypes(include=['number']).columns

# Normalize only numeric columns
odf[numeric_cols] = (odf[numeric_cols] - odf[numeric_cols].min()) / (odf[numeric_cols].max() - odf[numeric_cols].min())

# Print the normalized dataset
print("\nNormalized datasetis created")

# The third part of your code
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume 'target_column' is your target variable
X = odf.drop('Class', axis=1)
y = odf['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop non-numeric columns before normalization
non_numeric_cols = X.select_dtypes(exclude='number').columns
X_train_numeric = X_train.drop(columns=non_numeric_cols)
X_test_numeric = X_test.drop(columns=non_numeric_cols)

# Initialize a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize RFE
rfe = RFE(model, n_features_to_select=10)  # Choose the number of features you want to select

# Fit RFE
fit = rfe.fit(X_train_numeric, y_train)

# Get selected features
selected_features_rfe = X_train_numeric.columns[fit.support_]

# Train a model with the selected features
model_rfe = RandomForestClassifier(n_estimators=100, random_state=42)
model_rfe.fit(X_train[selected_features_rfe], y_train)

# Make predictions on the test set
y_pred_rfe = model_rfe.predict(X_test[selected_features_rfe])

# Evaluate the model
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print("Accuracy with RFE selected features:", accuracy_rfe)


# The fourth part of your code
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assume 'target_column' is your target variable
X = odf.drop('Class', axis=1)
y = odf['Class']

# Convert any categorical variables to numerical using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SelectKBest with chi-squared test
k_best = SelectKBest(chi2, k=10)  # Choose the number of features you want to select

# Fit SelectKBest
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Get selected features indices
selected_feature_indices = k_best.get_support(indices=True)

# Train a model with the selected features
model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
model_selected.fit(X_train_kbest, y_train)

# Make predictions on the test set
y_pred_selected = model_selected.predict(X_test_kbest)

# Evaluate the model
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with SelectKBest selected features:", accuracy_selected)


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'your_file.csv' with the path to your CSV file
file_path = 'C:\\Users\\smrit\\OneDrive\\Desktop\\sem6\\CSA\\Obfuscated-MalMem2022.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
# print(df)
print("3rd task")
cols_sel = ['pslist.nppid', 'dlllist.avg_dlls_per_proc', 'ldrmodules.not_in_mem',
                      'psxview.not_in_deskthrd', 'svcscan.kernel_drivers', 'callbacks.ncallbacks','Class']

# Randomly select 15 rows and specific columns
r15 = df.sample(n=15)[cols_sel]
# Display the selected data


print(r15)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("6th task")
print('Scatter plot')
# Extract the last two numerical attributes
last_two_attributes = r15.iloc[:, -2:]
# Scatter plot
plt.scatter(last_two_attributes.iloc[:, 0], last_two_attributes.iloc[:, 1])
plt.title('Scatter Plot for Last Two Numerical Attributes (Randomly Selected 15 Rows)')
plt.xlabel('Last Numerical Attribute')
plt.ylabel('Second to Last Numerical Attribute')
plt.show()

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print('7th task')
print('Histogram')
first_three_attributes = r15.iloc[:, :3]

# Plot histograms for the first three numerical attributes
first_three_attributes.plot.hist(alpha=0.5, bins=20, figsize=(12, 6), layout=(1, 3), sharex=True, sharey=True)
plt.suptitle('Histograms for First Three Numerical Attributes (Randomly Selected 15 Rows)')
plt.show()
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print('8th task')
print('Analyze first attribute')
attribute_1 = r15.iloc[:, 0]

# Display descriptive statistics
print("Descriptive Statistics for Attribute 1:")
print(attribute_1.describe())

# Plot a histogram
plt.hist(attribute_1, bins=20, edgecolor='black')
plt.title('Histogram for Attribute 1 (Randomly Selected 15 Rows)')
plt.xlabel('Attribute 1 Values')
plt.ylabel('Frequency')
plt.show()
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print('9th task')
print('Box plot')
attribute_1 = r15.iloc[:, 0]

# Create a box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=attribute_1)
plt.title('Box Plot for Attribute 1 (Randomly Selected 15 Rows)')
plt.xlabel('Attribute 1')
plt.show()
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print('10th task')
print('Scatter plots')
# Selecting two numerical attributes for scatter plots
attribute1 = 'callbacks.ncallbacks'
attribute2 = 'dlllist.avg_dlls_per_proc'
class_variable = 'Class'

colors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #Variable colors for different 15 variables

# Create a scatter plot using matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(r15[class_variable], r15[attribute1], c=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], cmap='viridis', alpha=0.5)
plt.xlabel(class_variable)
plt.ylabel(attribute1)
plt.title(f'Scatter Plot: {attribute1} vs {class_variable}')

plt.subplot(2, 2, 2)
plt.scatter(r15[class_variable], r15[attribute2], c=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], cmap='viridis', alpha=0.5)
plt.xlabel(class_variable)
plt.ylabel(attribute2)
plt.title(f'Scatter Plot: {attribute2} vs {class_variable}')

# Create a scatter plot using seaborn
plt.subplot(2, 2, 3)
sns.scatterplot(x=class_variable, y=attribute1, hue=class_variable, data=r15, palette='viridis', alpha=0.7)
plt.title(f'Scatter Plot(seaborn): {attribute1} vs {class_variable}')

plt.subplot(2, 2, 4)
sns.scatterplot(x=class_variable, y=attribute2, hue=class_variable, data=r15, palette='viridis', alpha=1)
plt.title(f'Scatter Plot(seaborn): {attribute2} vs {class_variable}')

plt.tight_layout()
#plt.show()

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("11th task")
# Assuming r15 is your DataFrame with the selected rows and columns

# Extract the first numerical attribute
attribute_1 = r15.iloc[:, 0]

# Box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x=attribute_1)
plt.title('Box Plot for Attribute 1 (Randomly Selected 15 Rows)')
plt.xlabel('Attribute 1 Values')
plt.show()
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
