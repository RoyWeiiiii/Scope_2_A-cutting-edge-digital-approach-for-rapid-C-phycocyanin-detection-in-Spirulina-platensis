import os
import logging
import numpy as np
import cv2
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import learning_curve

########### Define for CI plot ############
def compute_r2_confidence_interval(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    r2_samples = []
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        r2_samples.append(r2_score(y_true[indices], y_pred[indices]))
    ci_lower = np.percentile(r2_samples, 2.5)
    ci_upper = np.percentile(r2_samples, 97.5)
    return np.mean(r2_samples), ci_lower, ci_upper

def plot_r2_confidence_intervals(model_r2s_with_ci,  save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(model_r2s_with_ci.keys())
    means = [val[0] for val in model_r2s_with_ci.values()]
    lowers = [val[0] - val[1] for val in model_r2s_with_ci.values()]
    uppers = [val[2] - val[0] for val in model_r2s_with_ci.values()]
    ci_widths = [lo + hi for lo, hi in zip(lowers, uppers)]
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(labels)))

    # Print CI summary
    print("\n=== R² Confidence Intervals (Validation) ===")
    for label, mean, lo, hi in zip(labels, means, lowers, uppers):
        lo_val, hi_val = mean - lo, mean + hi
        print(f"{label}: R² = {mean:.4f} | CI: [{lo_val:.4f}, {hi_val:.4f}]")

    # Plot
    plt.figure(figsize=(12, 7))
    x = np.arange(len(labels))
    bar_width = 0.6
    bars = plt.bar(x, means, yerr=[lowers, uppers], capsize=6,
                   color=colors, edgecolor='black', width=bar_width)

    # Annotate bars
    for i, (mean, lo, hi) in enumerate(zip(means, lowers, uppers)):
        lo_val = mean - lo
        hi_val = mean + hi
        plt.text(i, hi_val + 0.002, f"R²={mean:.3f}\n[{lo_val:.3f}, {hi_val:.3f}]",
                 ha='center', va='bottom', fontsize=10)

    # Dynamic Y-limit with padding
    upper_y = max(mean + hi for mean, hi in zip(means, uppers))
    plt.ylim(0, upper_y + 0.01)

    plt.xticks(x, labels, rotation=30)
    plt.ylabel("R² Score")
    plt.title("R² Scores with 95% Confidence Intervals (Bootstrapped)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    

########
logger = logging.getLogger(name='Logger')
logger.setLevel(logging.INFO)

class BiomassDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.cpc_concentrations = []
        self.DAYS = ['Day_2', 'Day_4', 'Day_6', 'Day_8', 'Day_10', 'Day_12']
        self.DEVICES = ['Iphone_13_Pro_Max']
        self.CONDITIONS = ['Light_disturbance']
        self.CROP_BATCHES = ['Batch_1_crop', 'Batch_2_crop', 'Batch_3_crop']
        self.DESIRED_PIXEL_SIZE = (250, 250)
        self.CPC_CONCENTRATION_MAP = {

# # Iphone_Covered
    # ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.0833,
    # ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.0760,
    # ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.0766,
    # ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1538,
    # ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1175,
    # ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1734,
    # ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.2291,
    # ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2193,
    # ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.2193,
    # ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.4220,
    # ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.4458,
    # ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.4261,
    # ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.3856,
    # ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2757,
    # ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.3795,
    # ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1551,
    # ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1710,
    # ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1539,

# # Iphone_No_Covered
    ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.0766,
    ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.1734,
    ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.2193,
    ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.4261,
    ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.3795,
    ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.1539,

# Camera_Covered
    # ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.0833,
    # ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.0760,
    # ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.0766,
    # ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.1538,
    # ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.1175,
    # ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.1734,
    # ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.2291,
    # ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.2193,
    # ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.2193,
    # ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.4220,
    # ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.4458,
    # ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.4261,
    # ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.3856,
    # ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.2757,
    # ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.3795,
    # ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.1551,
    # ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.1710,
    # ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.1539,

# Camera_No_Covered
    # ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.0833,
    # ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.0760,
    # ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.0766,
    # ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.1538,
    # ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.1175,
    # ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.1734,
    # ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.2291,
    # ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.2193,
    # ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.2193,
    # ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.4220,
    # ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.4458,
    # ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.4261,
    # ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.3856,
    # ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.2757,
    # ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.3795,
    # ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.1551,
    # ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.1710,
    # ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.1539,
    
        }
        self.__load_data()

    def __load_data(self):
        for day in self.DAYS:
            for device in self.DEVICES:
                for condition in self.CONDITIONS:
                    for crop_batch in self.CROP_BATCHES:
                        batch_dir = os.path.join(self.dataset_path, day, device, condition, crop_batch)
                        cpc_concentration = self.CPC_CONCENTRATION_MAP[(day, device, condition, crop_batch)]
                        for img_file in os.listdir(batch_dir):
                            img_path = os.path.join(batch_dir, img_file)
                            img = cv2.imread(img_path)
                            img = cv2.resize(img, self.DESIRED_PIXEL_SIZE).transpose(2, 0, 1) / 255.0
                            self.images.append(img)
                            self.cpc_concentrations.append(cpc_concentration)
        self.images = np.array(self.images, dtype=np.float32)
        self.cpc_concentrations = np.array(self.cpc_concentrations, dtype=np.float32)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.cpc_concentrations[index]

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()  # Flatten the feature map to prepare for SVM & XGBoost
        )

    def forward(self, x):
        return self.feature_extractor(x)

def plot_box_plots(model_residuals, title="Residuals Comparison Across Models", ylabel="Residuals (Actual - Predicted)", save_path=None):
    # Plot
    plt.figure(figsize=(10, 6))
    pd.DataFrame(model_residuals).boxplot()
    plt.title(title)
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save to file if provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📦 Residual boxplot saved to: {save_path}")
    plt.close()

# Define residual stdev barplot
# def plot_residual_stddev_barplot(std_devs, save_path=None):
#     plt.figure(figsize=(10, 5))
#     sns.barplot(x=list(std_devs.keys()), y=list(std_devs.values()), palette='crest', legend=False)
#     plt.title("Standard Deviation of Residuals")
#     plt.ylabel("Residual Std Dev")
#     plt.xticks(rotation=30)
#     plt.grid(True)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()

#     plt.close()

# Feature extraction function
def extract_features(data_loader, model, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            features.append(model(images).cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)

#Corrected visualization function - Scatter plots
# def plot_actual_vs_predicted(actual, predicted, model_name, save_dir=None):
#     plt.figure(figsize=(10, 6))
#     plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=2, label='Perfect Prediction', alpha=0.7)
#     plt.scatter(actual, actual, color='red', alpha=0.7, label='Actual CPC Concentration', marker='o')
#     plt.scatter(actual, predicted, color='blue', alpha=0.7, label='Predicted CPC Concentration', marker='x')
#     plt.xlabel('Actual CPC Concentration')
#     plt.ylabel('Predicted CPC Concentration')
#     plt.title(f'{model_name} Prediction vs Actual')
#     plt.legend()
#     plt.grid(True)

#     if save_dir:
#         filename = f"{model_name.replace(' ', '_')}_scatter.png"
#         plt.savefig(os.path.join(save_dir, filename))
#     else:
#         plt.show()
#     plt.close()

# Learning curve function
# def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring="r2", save_path=None):
#     train_sizes, train_scores, validation_scores = learning_curve(
#         estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
#     )
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score", alpha=0.8)
#     plt.plot(train_sizes, np.mean(validation_scores, axis=1), 's-', label="Validation Score", alpha=0.8)
#     plt.fill_between(train_sizes, 
#                      np.mean(train_scores, axis=1) - np.std(train_scores, axis=1), 
#                      np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
#     plt.fill_between(train_sizes, 
#                      np.mean(validation_scores, axis=1) - np.std(validation_scores, axis=1), 
#                      np.mean(validation_scores, axis=1) + np.std(validation_scores, axis=1), alpha=0.1)
#     plt.xlabel("Training Set Size")
#     plt.ylabel(scoring.upper())
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)

#     if save_path:
#         plt.savefig(save_path)
#     else:
#         plt.show()
#     plt.close()

# Main workflow
def main():
    # Paths and device setup
    dataset_path = "D:\CodingProjects\machine_learning\Experiment_3\CNN_CPC_Regressor"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_r2s_with_ci = {}   # for R² CI tracking
    model_residuals = {}     # for residuals for boxplot
    model_scores = {}       # to record all model scores then store into csv file
    std_devs = {}

    # Paths to save all visualisations (TOGGLE)
    fig_export_folder = r"D:\CodingProjects\machine_learning\Experiment_3\Hybrid_Ensemble_results_Revision_2\SP-LD-CPC OK"
    os.makedirs(fig_export_folder, exist_ok=True)

    # Load dataset
    dataset = BiomassDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    
    print(f"Total data samples: {len(dataset)}")
    print(f"Training data samples: {len(train_dataset)}")
    print(f"Testing data samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize CNN
    cnn_model = CNNFeatureExtractor().to(device)

    # Extract features
    train_features, train_labels = extract_features(train_loader, cnn_model, device)
    test_features, test_labels = extract_features(test_loader, cnn_model, device)

    print(f"Number of features extracted: {train_features.shape[1]}")
    print(f"Training feature shape: {train_features.shape}")
    print(f"Testing feature shape: {test_features.shape}")

    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Models
    models = {
        "SVM": SVR(C=100, kernel='rbf', gamma='auto', epsilon=0.01),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=6, subsample=1.0, colsample_bytree=1.0, reg_alpha=2.0, reg_lambda=2.0 
        ) # decreased the learning rate from 0.3 to 0.1, reg_alpha from 0 to 2.0, alpha & lambda from 1.0 to 2.0 to prevent overfitting
    }

    for model_name, model in models.items():
        model.fit(train_features, train_labels)
        preds = model.predict(test_features)
        residuals = test_labels - preds
        model_residuals[model_name] = residuals
        print(f"{model_name} - Test Residuals Mean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}")

        # Save scatter and learning curve for base model
        # plot_actual_vs_predicted(test_labels, preds, model_name, save_dir=fig_export_folder)

        # plot_learning_curve(
        #     model, train_features, train_labels,
        #     title=f"{model_name} Learning Curve",
        #     save_path=os.path.join(fig_export_folder, f"{model_name}_learning_curve.png")
        # )

        # ✅ Compute R² CI for base model
        mean_r2, ci_lower, ci_upper = compute_r2_confidence_interval(test_labels, preds)
        model_r2s_with_ci[f"Base: {model_name}"] = (mean_r2, ci_lower, ci_upper)

        # Training and testing metrics
        for label, features, y in zip(["Train", "Test"], [train_features, test_features], [train_labels, test_labels]):
            preds = model.predict(features)
            mae = mean_absolute_error(y, preds)
            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, preds)

            if f'Base: {model_name}' not in model_scores:
                model_scores[f'Base: {model_name}'] = {}

            model_scores[f'Base: {model_name}'][label] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }

        # ✅ Residual StdDev (after test predictions!)
        test_preds = model.predict(test_features)
        residuals = test_labels - test_preds
        std_dev = np.std(residuals)
        model_scores[f'Base: {model_name}']["Residual StdDev"] = std_dev

        # Plot predictions and learning curve V1 OR(TOGGLE - print here)
        # plot_actual_vs_predicted(test_labels, model.predict(test_features), model_name)
        # plot_learning_curve(model, train_features, train_labels, title=f"{model_name} Learning Curve")
        
    meta_models = {
        'RidgeCV': RidgeCV(),
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(max_depth = 3),
        'RandomForest': RandomForestRegressor(n_estimators= 300, max_depth = 3),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=1.0, colsample_bytree=1.0, reg_alpha=2.0, reg_lambda=2.0),
        'SVR': SVR(C=100, kernel='rbf', gamma='auto', epsilon=0.01),
    }

    for meta_name, meta_model in meta_models.items():
        stacking_model = StackingRegressor(
            estimators=[('svm', models['SVM']), ('xgb', models['XGBoost'])], final_estimator=meta_model,
        )
        stacking_model.fit(train_features, train_labels)

        for label, features, y in zip(["Train", "Test"], [train_features, test_features], [train_labels, test_labels]):
            preds = stacking_model.predict(features)
            mae = mean_absolute_error(y, preds)
            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, preds)
            print(f"{meta_name} Stacking Model ({label}) - MAE: {mae:}, MSE: {mse:}, RMSE: {rmse:}, R2: {r2:}")

            if f'Stacking: {meta_name}' not in model_scores:
                model_scores[f'Stacking: {meta_name}'] = {}

            model_scores[f'Stacking: {meta_name}'][label] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2
            }
        
        # CI
        if label == "Test":
            ...
            mean_r2, ci_lower, ci_upper = compute_r2_confidence_interval(y, preds)
            model_r2s_with_ci[f'Stacking: {meta_name}'] = (mean_r2, ci_lower, ci_upper)

            residuals = y - preds
            model_residuals[f"Stacking: {meta_name}"] = residuals
            std_devs[f"Stacking: {meta_name}"] = np.std(residuals)

        # if label == "Test":  # Scatter plot for Test data
        #     plot_actual_vs_predicted(y, preds, f"{meta_name} Stacking Model",save_dir=fig_export_folder) # Toggle for scatterplot

        # Plot learning curve
        # plot_learning_curve(stacking_model, train_features, train_labels, title=f"{meta_name} Stacking Model Learning Curve")  #Toggle for learning plot

        # Save learning curve for current meta model
        # plot_learning_curve(stacking_model, train_features, train_labels,
        #                     title=f"Meta-{meta_name} Learning Curve",
        #                     save_path=os.path.join(fig_export_folder, f"Meta-{meta_name}_learning_curve.png"))
    
    model_residuals = {}

    # Base model boxplot
    for model_name, model in models.items():
        model.fit(train_features, train_labels)
        preds = model.predict(test_features)
        residuals = test_labels - preds

        model_residuals[model_name] = residuals
        std_devs[f"Base: {model_name}"] = np.std(residuals)

        print(f"{model_name} - Test Residuals Mean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}")
    
    # Meta models boxplot
    for meta_name, meta_model in meta_models.items():
        stacking_model = StackingRegressor(
            estimators=[('svm', models['SVM']), ('xgb', models['XGBoost'])], final_estimator=meta_model
        )
        stacking_model.fit(train_features, train_labels)
        preds = stacking_model.predict(test_features)
        residuals = test_labels - preds
        model_residuals[f"Meta-{meta_name}"] = residuals
        std_devs[f"Meta-{meta_name}"] = np.std(residuals)
        print(f"Meta-{meta_name} - Test Residuals Mean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}")
    
    # plot_box_plots(model_residuals)

    # Save the residual boxplot
    plot_box_plots(model_residuals, save_path=os.path.join(fig_export_folder, "residual_boxplot.png"))
    model_residuals = {}

    # Plot the confidence interval V1 here OR (TOGGLE)
    # plot_r2_confidence_intervals(model_r2s_with_ci)

    # Save the confidence interval V2 into folder (TOGGLE)
    # plot_r2_confidence_intervals(model_r2s_with_ci, save_path=os.path.join(fig_export_folder, 'r2_confidence_intervals.png'))
    
    # === Residual Std Dev Barplot ===
    # plt.figure(figsize=(10, 5))
    # sns.barplot(x=list(std_devs.keys()), y=list(std_devs.values()), palette='crest', legend=False)
    # plt.title("Standard Deviation of Residuals")
    # plt.ylabel("Residual Std Dev")
    # plt.xticks(rotation=30)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(fig_export_folder, "residual_stddev_barplot.png"))
    # plt.close()

    # === Export all model metrics to CSV ===
    all_model_results = []
    for model_name, metrics in model_scores.items():
        base_key = f"Base: {model_name}"

        # row = {
        #     "Model": base_key,
        #     "Train MAE": model_scores[base_key]["Train"]["MAE"],
        #     "Train MSE": model_scores[base_key]["Train"]["MSE"],
        #     "Train RMSE": model_scores[base_key]["Train"]["RMSE"],
        #     "Train R2": model_scores[base_key]["Train"]["R2"],
        #     "Test MAE": model_scores[base_key]["Test"]["MAE"],
        #     "Test MSE": model_scores[base_key]["Test"]["MSE"],
        #     "Test RMSE": model_scores[base_key]["Test"]["RMSE"],
        #     "Test R2": model_scores[base_key]["Test"]["R2"],
        #     "Test R2 CI": model_r2s_with_ci[base_key][0],
        #     "Test R2 CI Lower": model_r2s_with_ci[base_key][1],
        #     "Test R2 CI Upper": model_r2s_with_ci[base_key][2],
        #     "Residual StdDev": round(std_devs[base_key], 6) if base_key in std_devs else "N/A"
        # }

        # all_model_results.append(row)

        row = {
            "Model": model_name,
            "Train MAE": round(metrics['Train']['MAE'], 10),
            "Train MSE": round(metrics['Train']['MSE'], 10),
            "Train RMSE": round(metrics['Train']['RMSE'], 10),
            "Train R2": round(metrics['Train']['R2'], 10),
            "Test MAE": round(metrics['Test']['MAE'], 10),
            "Test MSE": round(metrics['Test']['MSE'], 10),
            "Test RMSE": round(metrics['Test']['RMSE'], 10),
            "Test R2": round(metrics['Test']['R2'], 10),
            "Test R2 CI Lower": round(model_r2s_with_ci[model_name][1], 10) if model_name in model_r2s_with_ci else "N/A",
            "Test R2 CI Upper": round(model_r2s_with_ci[model_name][2], 10) if model_name in model_r2s_with_ci else "N/A",
            "Residual StdDev": round(std_devs[model_name], 10) if model_name in std_devs else "N/A",
        }
        all_model_results.append(row)

    # === Save results to specified folder ===
    results_df = pd.DataFrame(all_model_results)

    csv_folder_path = r"D:\CodingProjects\machine_learning\Experiment_3\Hybrid_Ensemble_results_Revision_2\SP-LD-CPC OK"
    os.makedirs(csv_folder_path, exist_ok=True)

    csv_filename = "SP-LD-CPC.csv"
    csv_full_path = os.path.join(csv_folder_path, csv_filename)

    # results_df.to_csv(csv_full_path, index=False, float_format="%.6f")
    results_df.to_csv(csv_full_path, index=False)
    print(f"✅ CSV results saved to: {csv_full_path}")

if __name__ == "__main__":
    main()