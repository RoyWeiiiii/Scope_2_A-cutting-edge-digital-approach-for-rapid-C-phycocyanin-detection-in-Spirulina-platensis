import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data from CSV
data = pd.read_csv(
    "D:\CodingProjects\machine_learning\Experiment_3\Data_Colour_Index_Normalised\RGB\\"
    "RGB_Phone_covered_Biomass.csv"
)

# Define features (X)
# Data colour index
X = data[['R','G','B','IndexRGB']].values
# X = data[['H','S','L','IndexHSL']].values
# X = data[['C','M','Y','K','IndexCMYK']].values

# Data colour index & 'Day'
# X = data[['R','G','B','IndexRGB','Day']].values
# X = data[['H','S','L','IndexHSL','Day']].values
# X = data[['C','M','Y','K','IndexCMYK','Day']].values

# Data colour index, 'Day', & 'Abs'
# X = data[['R','G','B','IndexRGB','Day','Abs']].values
# X = data[['Abs', 'H','S','L','IndexHSL','Day']].values
# X = data[['Abs', 'C','M','Y','K','IndexCMYK','Day']].values

# Define target variable (y)
y = data['CPC'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM model
svm = SVR()

# Define hyperparameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf'],
              'gamma': ['scale'],
              'epsilon': [0.01],
              }

# Create GridSearchCV object with verbose=2 to show progress
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, 
                           scoring='neg_mean_squared_error', verbose=2, refit=True, n_jobs=-1)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_hyperparameters = grid_search.best_params_

print(f'Best Hyperparameters: {best_hyperparameters}')

# Get the best model
best_svm = grid_search.best_estimator_

# Train the best model on the entire training dataset
best_svm.fit(X_train, y_train)

# Make predictions on the testing dataset
y_pred = best_svm.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2) Score (Accuracy): {r2}')

# Visualization
plt.figure(figsize=(10, 6))
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Perfect Prediction', alpha=0.7)
plt.scatter(y_test, y_test, color='red', alpha=0.7, label='Actual CPC Concentration', marker='o')
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted CPC Concentration', marker='x')
plt.xlabel('Actual CPC Concentration')
plt.ylabel('Predicted CPC Concentration')
plt.title('RGB_Smartphone_Covered_ColourIndex')

# plt.text(0.1, 0.9, f'R-squared (Accuracy): {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.legend()
plt.grid(True)
plt.show()

