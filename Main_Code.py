import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# 1. LOAD DATA
df = pd.read_csv('predictive_maintenance.csv')

# 2. PREPROCESSING
# Drop UDI and Product ID as they are just unique identifiers
df = df.drop(['UDI', 'Product ID'], axis=1)

# Encode 'Type' (L, M, H quality tiers)
le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

# Separate Features and Target
X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1) # Drop failure subtypes
y = df['Machine failure']

# 3. ADDRESSING CLASS IMBALANCE
# Most machines don't fail, so we create synthetic failure data points
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. RANDOM FOREST WITH GRIDSEARCH (Optimization)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='recall')
grid_search.fit(X_train, y_train)

# 6. EVALUATION
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Parameters:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. VISUALIZATION: Feature Importance
importances = best_model.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance in Predicting Machine Failure')
plt.show()