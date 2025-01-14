import os
import logging
import numpy as np
import cv2
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

logger = logging.getLogger(name='Logger')
logger.setLevel(logging.INFO)

class BiomassDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.cpc_concentrations = []
        self.DAYS = ['Day_2', 'Day_4', 'Day_6', 'Day_8', 'Day_10', 'Day_12']
        self.DEVICES = ['Nikon_Z50']
        self.CONDITIONS = ['Light_disturbance']
        self.CROP_BATCHES = ['Batch_1_crop', 'Batch_2_crop', 'Batch_3_crop']
        self.DESIRED_PIXEL_SIZE = (250, 250)
        self.CPC_CONCENTRATION_MAP = {

# # Iphone_Covered
#     ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.0833,
#     ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.0760,
#     ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.0766,
#     ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1538,
#     ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1175,
#     ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1734,
#     ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.2291,
#     ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2193,
#     ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.2193,
#     ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.4220,
#     ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.4458,
#     ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.4261,
#     ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.3856,
#     ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2757,
#     ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.3795,
#     ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1551,
#     ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1710,
#     ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1539,

# # Iphone_No_Covered
#     ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.0833,
#     ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.0760,
#     ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.0766,
#     ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.1538,
#     ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.1175,
#     ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.1734,
#     ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.2291,
#     ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.2193,
#     ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.2193,
#     ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.4220,
#     ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.4458,
#     ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.4261,
#     ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.3856,
#     ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.2757,
#     ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.3795,
#     ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_1_crop'): 0.1551,
#     ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_2_crop'): 0.1710,
#     ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbance', 'Batch_3_crop'): 0.1539,

# # Camera_Covered
#     ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.0833,
#     ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.0760,
#     ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.0766,
#     ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.1538,
#     ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.1175,
#     ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.1734,
#     ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.2291,
#     ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.2193,
#     ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.2193,
#     ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.4220,
#     ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.4458,
#     ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.4261,
#     ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.3856,
#     ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.2757,
#     ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.3795,
#     ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.1551,
#     ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.1710,
#     ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.1539,

# Camera_No_Covered
    ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.0766,
    ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.1734,
    ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.2193,
    ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.4261,
    ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.3795,
    ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.1539,
    
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
    
def plot_box_plots(model_residuals):
    plt.figure(figsize=(10, 6))
    pd.DataFrame(model_residuals).boxplot()
    plt.title("Residuals Comparison Across Models")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.grid(True)
    plt.show()
    
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

#Corrected visualization function
def plot_actual_vs_predicted(actual, predicted, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'k--', lw=2, label='Perfect Prediction', alpha=0.7)
    plt.scatter(actual, actual, color='red', alpha=0.7, label='Actual CPC Concentration', marker='o')
    plt.scatter(actual, predicted, color='blue', alpha=0.7, label='Predicted CPC Concentration', marker='x')
    plt.xlabel('Actual CPC Concentration')
    plt.ylabel('Predicted CPC Concentration')
    plt.title(f'{model_name} Prediction vs Actual')
    plt.legend()
    plt.grid(True)
    plt.show()

# Learning curve function
def plot_learning_curve(estimator, X, y, title="Learning Curve", cv=5, scoring="r2"):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score", alpha=0.8)
    plt.plot(train_sizes, np.mean(validation_scores, axis=1), 's-', label="Validation Score", alpha=0.8)
    plt.fill_between(train_sizes, 
                     np.mean(train_scores, axis=1) - np.std(train_scores, axis=1), 
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    plt.fill_between(train_sizes, 
                     np.mean(validation_scores, axis=1) - np.std(validation_scores, axis=1), 
                     np.mean(validation_scores, axis=1) + np.std(validation_scores, axis=1), alpha=0.1)
    plt.xlabel("Training Set Size")
    plt.ylabel(scoring.upper())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main workflow
def main():
    # Paths and device setup
    dataset_path = "D:\CodingProjects\machine_learning\Experiment_3\CNN_CPC_Regressor"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


        # Training and testing metrics
        for label, features, y in zip(["Train", "Test"], [train_features, test_features], [train_labels, test_labels]):
            preds = model.predict(features)
            mae = mean_absolute_error(y, preds)
            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, preds)
            print(f"{model_name} ({label}) - MAE: {mae:}, MSE: {mse:}, RMSE: {rmse:}, R2: {r2:}")

        # Plot predictions and learning curve
        plot_actual_vs_predicted(test_labels, model.predict(test_features), model_name)
        plot_learning_curve(model, train_features, train_labels, title=f"{model_name} Learning Curve")

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
            estimators=[('svm', models['SVM']), ('xgb', models['XGBoost'])], final_estimator=meta_model
        )
        stacking_model.fit(train_features, train_labels)

        for label, features, y in zip(["Train", "Test"], [train_features, test_features], [train_labels, test_labels]):
            preds = stacking_model.predict(features)
            mae = mean_absolute_error(y, preds)
            mse = mean_squared_error(y, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, preds)
            print(f"{meta_name} Stacking Model ({label}) - MAE: {mae:}, MSE: {mse:}, RMSE: {rmse:}, R2: {r2:}")

        if label == "Test":  # Scatter plot for Test data
            plot_actual_vs_predicted(y, preds, f"{meta_name} Stacking Model")

        #Plot learning curve
        plot_learning_curve(stacking_model, train_features, train_labels, title=f"{meta_name} Stacking Model Learning Curve")
    
    model_residuals = {}
    
    def plot_box_plots(model_residuals):
        plt.figure(figsize=(10, 6))
        pd.DataFrame(model_residuals).boxplot()
        plt.title("Residuals Comparison Across Models for DC-LD-CPC")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.xticks(rotation=45)  # Tilt x-axis labels by 45 degrees
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent cutting off labels
        plt.show()

    # Base model boxplot
    for model_name, model in models.items():
        model.fit(train_features, train_labels)
        preds = model.predict(test_features)
        residuals = test_labels - preds
        model_residuals[model_name] = residuals
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
        print(f"Meta-{meta_name} - Test Residuals Mean: {np.mean(residuals):.4f}, Std: {np.std(residuals):.4f}")
    
    plot_box_plots(model_residuals)

if __name__ == "__main__":
    main()