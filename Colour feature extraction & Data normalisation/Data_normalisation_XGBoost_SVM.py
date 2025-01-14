####################### Min max scaler normalisation ##############
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Specify the path to the folder containing the input CSV file
input_folder = "E:\CodingProjects\machine_learning\Experiment_3\Data_Abs_Day_Colour_index\CMYK"

# Specify the filename of the input CSV file
input_file = "E:\CodingProjects\machine_learning\Experiment_3\Data_Abs_Day_Colour_index\CMYK\CMYK_Combine_All_Batch_Days_Phone_non_covered_CPC.csv"

# Specify the path to the folder where you want to save the normalized CSV file
output_folder = "E:\CodingProjects\machine_learning\Experiment_3\Data_Abs_Day_Colour_index_Normalised\CMYK"

# Read the CSV file into a Pandas DataFrame
input_file_path = os.path.join(input_folder, input_file)
df = pd.read_csv(input_file_path)

# Define the columns you want to normalize
# columns_to_normalize = ['R', 'G', 'B', 'IndexRGB','Abs']
# columns_to_normalize = ['H', 'S', 'L', 'IndexHSL', 'Abs']
columns_to_normalize = ['C', 'M', 'Y', 'K', 'IndexCMYK','Abs']

# Normalize the specified columns using Min-Max scaling
scaler = MinMaxScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Define the output file path
output_file = os.path.join(output_folder, 'CMYK_Phone_no_covered_CPC_Abs_Day.csv')

# Save the normalized DataFrame to the specified folder
df.to_csv(output_file, index=False)

print(f"Normalized DataFrame saved to {output_file}")

