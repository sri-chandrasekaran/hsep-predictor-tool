import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Read the data set from the file
X = pd.read_csv('/Users/moulichandrasekaran/Desktop/hsep-predictor/anonymous-msweb.csv')

# Replace missing values with mean imputation
X = X.replace('?', np.nan)
X = X.apply(pd.to_numeric, errors='coerce')

# Separate predictor class
y = X['Status']
X = X.drop('Status', axis=1)

# Convert the target variable to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check the unique values in y_encoded
print(np.unique(y_encoded))

# Define numeric features for imputation
numeric_features = X.select_dtypes(include=[np.number]).columns

# Create the transformer pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Apply the preprocessing steps
X_transformed = preprocessor.fit_transform(X)

# Convert the transformed array to a DataFrame with column names
X_transformed = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out(numeric_features))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.30, random_state=7)


# SVM Classifier Model #1: using all of the attributes in the dataset
clf = SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Experiment#1: accuracy score:', accuracy_score(y_test, y_pred))

# SVM Classifier Model #2: using attribute reduction through PCA 
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf_pca = SVC(kernel='rbf', gamma='scale')
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
print('Experiment#2: accuracy score:', accuracy_score(y_test, y_pred_pca))
