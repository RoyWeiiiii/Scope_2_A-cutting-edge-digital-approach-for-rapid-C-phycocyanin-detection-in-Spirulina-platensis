############################# Single model at a time with 95% CI plot & Standard deviation (Std) ######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data from CSV
# Biomass dataset
data = pd.read_csv(
  "D:\CodingProjects\machine_learning\Experiment_3\Data_Day_Colour_index_Normalised\HSL\HSL_Phone_covered_Biomass_Day.csv",
#   "D:\CodingProjects\machine_learning\Experiment_3\Data_Day_Colour_index_Normalised\HSL\HSL_Phone_no_covered_Biomass_Day.csv",
#   "D:\CodingProjects\machine_learning\Experiment_3\Data_Day_Colour_index_Normalised\HSL\HSL_Camera_covered_Biomass_Day.csv",
#   "D:\CodingProjects\machine_learning\Experiment_3\Data_Day_Colour_index_Normalised\HSL\HSL_Camera_no_covered_Biomass_Day.csv",

)

# CPC dataset
# data = pd.read_csv(
#     # "D:\CodingProjects\machine_learning\Experiment_3\Data_Colour_Index_Normalised\RGB\RGB_Phone_covered_CPC.csv",
#     # "D:\CodingProjects\machine_learning\Experiment_3\Data_Colour_Index_Normalised\RGB\RGB_Phone_no_covered_CPC.csv",
#     # "D:\CodingProjects\machine_learning\Experiment_3\Data_Colour_Index_Normalised\RGB\RGB_Camera_covered_CPC.csv"
#     # "D:\CodingProjects\machine_learning\Experiment_3\Data_Colour_Index_Normalised\RGB\RGB_Camera_no_covered_CPC.csv",

# )

# Define features (X)
# Data colour index
# X = data[['R','G','B','IndexRGB']].values
# X = data[['H','S','L','IndexHSL']].values
# X = data[['C','M','Y','K','IndexCMYK']].values

# Data colour index & 'Day'
# X = data[['R','G','B','IndexRGB','Day']].values
X = data[['H','S','L','IndexHSL','Day']].values
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
param_grid = {'C': [100],
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
plt.title('2V-HSL-SP-C-Biomass')

# plt.text(0.1, 0.9, f'R-squared (Accuracy): {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.legend()
plt.grid(True)
plt.show()

# === Bootstrap 95% CI for R² ===
n_bootstrap = 1000
rng = np.random.default_rng(seed=42)
r2_samples = []

for _ in range(n_bootstrap):
    indices = rng.choice(len(y_test), len(y_test), replace=True)
    r2_sample = r2_score(y_test[indices], y_pred[indices])
    r2_samples.append(r2_sample)

ci_lower = np.percentile(r2_samples, 2.5)
ci_upper = np.percentile(r2_samples, 97.5)

# === Compute residual standard deviation ===
residuals = y_test - y_pred
residual_std = np.std(residuals)

# === Display results ===
print(f"95% CI for R²: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Residual Std Dev: {residual_std:.4f}")

# === Plot CI as error bar ===
plt.figure(figsize=(6, 5))
mean = r2
low = mean - ci_lower
high = ci_upper - mean

plt.bar(['SVM'], [mean], yerr=[[low], [high]], capsize=10, color='skyblue', edgecolor='black')
plt.text(0, ci_upper + 0.0003,
         f"R²={mean:.4f}\n95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\nStd Dev: {residual_std:.4f}",
         ha='center', va='bottom', fontsize=9)

plt.ylim(ci_lower - 0.002, ci_upper + 0.005)
plt.title("R² Score with 95% CI and Residual Std Dev")
plt.ylabel("R² Score")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



