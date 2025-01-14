import os
import logging
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

logger = logging.getLogger(name='Logger')
logger.setLevel(logging.INFO)

class BiomassDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.cpc_concentrations = []
        self.DAYS = ['Day_2', 'Day_4', 'Day_6', 'Day_8', 'Day_10', 'Day_12']
        self.DEVICES = ['Iphone_13_Pro_Max']
        self.CONDITIONS = ['Covered']
        self.CROP_BATCHES = ['Batch_1_crop', 'Batch_2_crop', 'Batch_3_crop']
        self.DESIRED_PIXEL_SIZE = (250, 250)
        self.CPC_CONCENTRATION_MAP = {

# Iphone_Covered
    ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.0766,
    ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1734,
    ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.2193,
    ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.4261,
    ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.3795,
    ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1539,

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

# # Camera_No_Covered
#     ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.0833,
#     ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.0760,
#     ( 'Day_2', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.0766,
#     ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.1538,
#     ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.1175,
#     ( 'Day_4', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.1734,
#     ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.2291,
#     ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.2193,
#     ( 'Day_6', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.2193,
#     ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.4220,
#     ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.4458,
#     ( 'Day_8', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.4261,
#     ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.3856,
#     ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.2757,
#     ( 'Day_10', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.3795,
#     ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_1_crop'): 0.1551,
#     ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_2_crop'): 0.1710,
#     ( 'Day_12', 'Nikon_Z50','Light_disturbance', 'Batch_3_crop'): 0.1539,
    
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

class CNNRegressionModel(nn.Module):
    def __init__(self):
        super(CNNRegressionModel, self).__init__()
        self.model = nn.Sequential(
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
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(6400, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        output = self.model(x)
        return output

def train_once():
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATASET_DIR = "D:\CodingProjects\machine_learning\Experiment_3\CNN_Biomass_regressor"
    logger.info('Load Dataset')
    dataset = BiomassDataset(DATASET_DIR)

    logger.info('Prepare Train and Test Dataset')
    training_dataset_length = math.ceil(len(dataset) * 0.8)
    testing_dataset_length = len(dataset) - training_dataset_length

    train_dataset, test_dataset = random_split(dataset, [training_dataset_length, testing_dataset_length])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    logger.info('Prepare Model, Optimizer and Criterion')
    model = CNNRegressionModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    mae_loss = MeanAbsoluteError().to(DEVICE)
    r2_loss = R2Score().to(DEVICE)
    rmse_loss = MeanSquaredError(squared=False).to(DEVICE) # Set `squared=False` for RMSE

    # Store Metrics
    loss_dict = {}
    loss_dict['mse_loss'] = {}
    loss_dict['mse_loss']['train'] = []
    loss_dict['mse_loss']['test'] = []
    loss_dict['r2_loss'] = {}
    loss_dict['r2_loss']['train'] = []
    loss_dict['r2_loss']['test'] = []
    loss_dict['mae_loss'] = {}
    loss_dict['mae_loss']['train'] = []
    loss_dict['mae_loss']['test'] = []
    loss_dict['rmse_loss'] = {}
    loss_dict['rmse_loss']['train'] = []
    loss_dict['rmse_loss']['test'] = []

    progress_bar = tqdm(range(NUM_EPOCHS))
    for epoch in progress_bar:
        model.train()
        train_mse_loss = 0
        train_r2_loss = 0
        train_mae_loss = 0
        train_rmse_loss = 0
        for (images, cpc_concentrations) in train_dataloader:
            images, cpc_concentrations = images.to(DEVICE), cpc_concentrations.to(DEVICE)
            cpc_concentrations = cpc_concentrations
            outputs = model(images)
            outputs = outputs.squeeze(1)
            mse_loss = criterion(outputs, cpc_concentrations)

            optimizer.zero_grad()  # Clear gradients
            mse_loss.backward()        # Compute gradients
            optimizer.step() 

            # Track loss (training)
            train_mse_loss += mse_loss.item()
            train_mae_loss += mae_loss(outputs, cpc_concentrations).item()
            train_r2_loss += r2_loss(outputs, cpc_concentrations).item()
            train_rmse_loss += rmse_loss(outputs, cpc_concentrations).item()

            progress_bar.set_description(f"Mode: Train | Epoch: {epoch} | MSE Loss: {train_mse_loss} | MAE Loss: {train_mae_loss} | R2 Loss: {train_r2_loss} | RMSE Loss: {train_rmse_loss}")


        train_mse_loss /= len(train_dataloader)
        train_mae_loss /= len(train_dataloader)
        train_r2_loss /= len(train_dataloader)
        train_rmse_loss /= len(train_dataloader)
        loss_dict['mse_loss']['train'].append(train_mse_loss)
        loss_dict['mae_loss']['train'].append(train_mae_loss)
        loss_dict['r2_loss']['train'].append(train_r2_loss)
        loss_dict['rmse_loss']['train'].append(train_rmse_loss)

        test_mse_loss = 0
        test_r2_loss = 0
        test_mae_loss = 0
        test_rmse_loss = 0
        model.eval()
        with torch.no_grad():
            for (images, cpc_concentrations) in test_dataloader:
                images, cpc_concentrations = images.to(DEVICE), cpc_concentrations.to(DEVICE)
                cpc_concentrations = cpc_concentrations.unsqueeze(1)
                outputs = model(images)
                mse_loss = criterion(outputs, cpc_concentrations)

                # Track loss (testing)
                test_mse_loss += mse_loss.item()
                test_mae_loss += mae_loss(outputs, cpc_concentrations).item()
                test_r2_loss += r2_loss(outputs, cpc_concentrations).item()
                test_rmse_loss += rmse_loss(outputs, cpc_concentrations).item()
                progress_bar.set_description(f"Mode: Train | Epoch: {epoch} | MSE Loss: {test_mse_loss} | MAE Loss: {test_mae_loss} | R2 Loss: {test_r2_loss} | RMSE Loss: {test_rmse_loss}")

            test_mse_loss /= len(test_dataloader)
            test_mae_loss /= len(test_dataloader)
            test_r2_loss /= len(test_dataloader)
            test_rmse_loss /= len(test_dataloader)
            loss_dict['mse_loss']['test'].append(test_mse_loss)
            loss_dict['mae_loss']['test'].append(test_mae_loss)
            loss_dict['r2_loss']['test'].append(test_r2_loss)
            loss_dict['rmse_loss']['test'].append(test_rmse_loss)

    SAVE_PATH = os.path.join('experiment_3_charts')

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save Dataframe
    # Plot loss curves
    formatted_loss_dict = {}
    for metric_key in loss_dict:
        train_losses = loss_dict[metric_key]['train']
        test_losses = loss_dict[metric_key]['test']
        formatted_loss_dict[f'train_{metric_key}'] = train_losses
        formatted_loss_dict[f'test_{metric_key}'] = test_losses
        # Plot loss curves
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"{metric_key.upper()} Training and Validation Loss")
        plt.savefig(os.path.join(SAVE_PATH, f"{metric_key}_loss_chart.png"))
        plt.close()

    
    df = pd.DataFrame(formatted_loss_dict)
    df.to_csv(os.path.join(SAVE_PATH, 'loss_df.csv'), index=False)


train_once()