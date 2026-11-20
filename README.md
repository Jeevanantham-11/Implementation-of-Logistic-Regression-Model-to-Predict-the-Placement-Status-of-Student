# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import Required Libraries

Step 2: Load the Dataset

Step 3: Copy Data & Drop Unwanted Columns

Step 4: Check Data Quality

Step 5: Encode Categorical Variables

Step 6: Define Features (X) and Target (y)

Step 7: Split into Training and Testing Sets

Step 8: Build and Train Logistic Regression Model

Step 9: Make Predictions

Step 10: Evaluate the Model

Step 11: Predict for a New Student

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Jeevanantham C
RegisterNumber: 212225240059

# 1. Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Load the Dataset
# NOTE: Change the file path if your CSV is in a different location
data = pd.read_csv("Placement_Data.csv")

# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())
```

<img width="597" height="252" alt="image" src="https://github.com/user-attachments/assets/845a456b-c126-40e2-affe-cfbf987904e6" />

```
# 3. Create a Copy and Drop Unwanted Columns
data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

```
<img width="602" height="257" alt="image" src="https://github.com/user-attachments/assets/13742d9f-7975-4403-a9a1-8e637481ea1f" />

```
# 4. Check for Missing and Duplicate Values
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

```

<img width="378" height="321" alt="image" src="https://github.com/user-attachments/assets/4b33b5ee-7774-4f81-a7be-1ec01c4bfc6d" />


```
# 5. Encode Categorical Variables using LabelEncoder
# Columns that are categorical (object type)
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())
```

<img width="605" height="252" alt="image" src="https://github.com/user-attachments/assets/1f0861e5-7311-4501-bea1-b883e5dcd17e" />

```
# 6. Define Features (X) and Target (y)
# X = all columns except 'status'
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())
```

<img width="602" height="392" alt="image" src="https://github.com/user-attachments/assets/6bc45e66-50cc-4550-9ef8-03cfa4185d85" />


```


# 7. Split the Dataset into Training and Testing Sets
# test_size=0.2 → 20% test data, 80% training data
# random_state=0 → same split every time (for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

```

<img width="247" height="102" alt="image" src="https://github.com/user-attachments/assets/3724f34b-9c6f-48fe-a242-50c6837b2c7c" />


```

# 8. Create and Train the Logistic Regression Model
# solver='liblinear' works well for small datasets
lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)
```
<img width="323" height="77" alt="image" src="https://github.com/user-attachments/assets/8f5b1143-f2df-4f73-a2a6-dc081f1cf38e" />


```
# 9. Make Predictions on the Test Set
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)
```
<img width="610" height="57" alt="image" src="https://github.com/user-attachments/assets/12b63e1a-d81c-442a-ac33-cc6262ce601f" />

```

# 10. Evaluate Model Performance
# Accuracy: percentage of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

<img width="450" height="205" alt="image" src="https://github.com/user-attachments/assets/3d4ad68b-d363-4fea-8cd0-76188f60a8b4" />


```

# 11. Predict Placement for a New Student
# Order of features must match X columns:
# ['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s',
#  'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p']

# Example student data (after encoding categorical values manually)
# NOTE: These categorical numeric codes depend on how LabelEncoder encoded them.
# Here we assume:
# gender: 1 (e.g., Male)
# ssc_p: 80
# ssc_b: 1
# hsc_p: 90
# hsc_b: 1
# hsc_s: 1
# degree_p: 90
# degree_t: 1
# workex: 0 (No work experience)
# etest_p: 85
# specialisation: 1
# mba_p: 85

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])

```

<img width="1378" height="75" alt="image" src="https://github.com/user-attachments/assets/288dcd12-156f-4652-9e64-7241bf9fd27e" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
