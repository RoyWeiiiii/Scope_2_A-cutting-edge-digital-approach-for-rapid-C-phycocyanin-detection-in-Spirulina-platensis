############## Working the best also with ROI (Multiple CPC concentration for each batch) ##################
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

# Function to calculate RGB, HSL, and CMYK values and color indices
def calculate_color_attributes(image_path, cpc_concentration):
    try:
        # Open the image using Pillow
        image = Image.open(image_path)

        if image is not None:
            # Convert the Pillow image to a numpy array
            image_array = np.array(image)

            # Calculate the dimensions of the middle 50% of the image
            height, width, _ = image_array.shape
            top = int(height * 0.25)
            bottom = int(height * 0.75)
            left = int(width * 0.25)
            right = int(width * 0.75)

            # Crop the image to the middle 50%
            cropped_image = image_array[top:bottom, left:right]

            # Display the original and cropped images
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(image_array)
            # plt.title("Original Image")
            # plt.axis('off')
            
            # plt.subplot(1, 2, 2)
            # plt.imshow(cropped_image)
            # plt.title("Cropped Image")
            # plt.axis('off')
            # plt.show()

            # Convert image to RGB color space
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Calculate RGB values
            rgb_values = rgb_image.mean(axis=(0, 1))

            # Calculate HSL values
            r, g, b = rgb_values / 255.0
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            l = (max_val + min_val) / 2

            if max_val == min_val:
                h = 0
                s = 0
            else:
                if l <= 0.5:
                    s = (max_val - min_val) / (max_val + min_val)
                else:
                    s = (max_val - min_val) / (2.0 - max_val - min_val)

                if max_val == r:
                    h = (g - b) / (max_val - min_val)
                elif max_val == g:
                    h = 2 + (b - r) / (max_val - min_val)
                else:
                    h = 4 + (r - g) / (max_val - min_val)

                h *= 60
                if h < 0:
                    h += 360

            # Convert S and L values to percentages
            s_percentage = int(s * 100)
            l_percentage = int(l * 100)

            hsl_values = (int(h), s_percentage, l_percentage)

            # Calculate CMYK values
            r, g, b = rgb_values
            r, g, b = r / 255.0, g / 255.0, b / 255.0

            k = 1 - max(r, g, b)
            if k == 1:
                c, m, y = 0, 0, 0
            else:
                c = int((1 - r - k) / (1 - k) * 100)
                m = int((1 - g - k) / (1 - k) * 100)
                y = int((1 - b - k) / (1 - k) * 100)

            # Convert C, M, Y, and K values to percentages
            k_percentage = int(k * 100)

            cmyk_values = (c, m, y, k_percentage)

            # Calculate color indices
            index_rgb = int(rgb_values[0] * 256 * 256 + rgb_values[1] * 256 + rgb_values[2])
            index_hsl = int(h * 101 * 101 + s * 101 + l)
            index_cmyk = int(c * 101 * 101 * 101 + m * 101 * 101 + y * 101 + k * 101)

            # Extract batch number from the image path
            batch_number = os.path.basename(os.path.dirname(image_path)).replace("Batch_", "")

        return {
            'Image': os.path.basename(image_path),
            'Batch_No.': batch_number,
            'Red (0-255)': int(rgb_values[0]),
            'Green (0-255)': int(rgb_values[1]),
            'Blue (0-255)': int(rgb_values[2]),
            'Hue (0-360)': int(h),
            'Saturation (0-100)': s_percentage,
            'Lightness (0-100)': l_percentage,
            'Cyan (0-100)': c,
            'Magenta (0-100)': m,
            'Yellow (0-100)': y,
            'Key (0-100)': k_percentage,
            'IndexRGB': index_rgb,
            'IndexHSL': index_hsl,
            'IndexCMYK': index_cmyk,
            'CPC concentration (mg/ml)': cpc_concentration,
            'Day': "12"  # Specify the desired Day value here
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Function to calculate color attributes for multiple images in a folder and store the results in a DataFrame
def calculate_color_attributes_for_folder(input_folder, cpc_concentration):
    # Get a list of all image files in the input folder
    image_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)
                  if filename.lower().endswith(('.JPG','.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    # Initialize an empty list to store the attributes for all images
    all_attributes = []

    # Loop through the image files and calculate attributes
    for image_path in image_files:
        attributes = calculate_color_attributes(image_path, cpc_concentration)
        if attributes:
            all_attributes.append(attributes)

    # Convert the list of attributes to a DataFrame
    df_batch = pd.DataFrame(all_attributes)

    return df_batch

if __name__ == "__main__":
    # Specify the input folders for the three batches of images and their respective CPC concentrations
    input_folders = [
        {
            'folder': r"E:\CodingProjects\machine_learning\Experiment_3\Spirulina_BG-11_CPC_images\Day_12\Nikon_Z50\Light_disturbance\Batch_1_crop",
            'cpc_concentration': '0.1551'
        },
        {
            'folder': r"E:\CodingProjects\machine_learning\Experiment_3\Spirulina_BG-11_CPC_images\Day_12\Nikon_Z50\Light_disturbance\Batch_2_crop",
            'cpc_concentration': '0.1710'
        },
        {
            'folder': r"E:\CodingProjects\machine_learning\Experiment_3\Spirulina_BG-11_CPC_images\Day_12\Nikon_Z50\Light_disturbance\Batch_3_crop",
            'cpc_concentration': '0.1539'
        }
    ]

    output_csv_file = r"E:\CodingProjects\machine_learning\Experiment_3\Spirulina_BG-11_CPC_images\Day_12\Nikon_Z50\Light_disturbance\Combine_Batch_Day_12_Camera_no_cover_160_CPC_IMG.csv"


    # Initialize an empty list to store DataFrames for all batches
    df_list = []

    # Loop through the input folders and calculate attributes for each batch
    for batch_info in input_folders:
        df = calculate_color_attributes_for_folder(batch_info['folder'], batch_info['cpc_concentration'])
        # Extract batch number from the folder name and replace it
        batch_number = os.path.basename(batch_info['folder']).replace("Batch_1_crop", "1") \
            .replace("Batch_2_crop", "2").replace("Batch_3_crop", "3")
        df['Batch_No.'] = batch_number
        df_list.append(df)
    
    # Combine DataFrames from all batches into one DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Rename the columns as specified
    column_rename = {
        'Avg_R': 'Red (0-255)',
        'Avg_G': 'Green (0-255)',
        'Avg_B': 'Blue (0-255)',
        'H': 'Hue (0-360)',
        'S': 'Saturation (0-100)',
        'L': 'Lightness (0-100)',
        'C': 'Cyan (0-100)',
        'M': 'Magenta (0-100)',
        'Y': 'Yellow (0-100)',
        'K': 'Key (0-100)',
    }
    combined_df = combined_df.rename(columns=column_rename)

    # Rearrange the columns in the desired order
    column_order = [
        'Red (0-255)', 'Green (0-255)', 'Blue (0-255)', 'Hue (0-360)', 'Saturation (0-100)',
        'Lightness (0-100)', 'Cyan (0-100)', 'Magenta (0-100)', 'Yellow (0-100)', 'Key (0-100)',
        'IndexRGB', 'IndexHSL', 'IndexCMYK', 'Image', 'Batch_No.', 'CPC concentration (mg/ml)', 'Day'
    ]
    combined_df = combined_df[column_order]

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_csv_file, index=False)
    
    print("Data processing completed.")

##################### To display the output for RGB, HSL, and CMYK ###########################
# from PIL import Image
# import colorsys
# import numpy as np
# import matplotlib.pyplot as plt

# # Open the RGB image
# rgb_image = Image.open("E:\CodingProjects\machine_learning\Experiment_3\Spirulina_BG-11_biomass_images\Day_2\Iphone_13_Pro_Max\Covered\Batch_1_crop\IMG_2176.JPG")

# # Convert to HSL
# hsl_image = rgb_image.convert('RGB')

# # Convert to CMYK
# cmyk_image = rgb_image.convert('CMYK')

# # Create subplots
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# # Display RGB image
# ax1.set_title('RGB')
# ax1.imshow(np.array(rgb_image))
# ax1.axis('off')

# # Convert RGB to HSL
# hsl_data = []
# for y in range(hsl_image.height):
#     row = []
#     for x in range(hsl_image.width):
#         r, g, b = hsl_image.getpixel((x, y))
#         h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
#         row.append((int(h * 360), int(s * 100), int(l * 100)))
#     hsl_data.append(row)

# # Display HSL image
# ax2.set_title('HSL')
# ax2.imshow(np.array(hsl_data, dtype=np.uint8))
# ax2.axis('off')

# # Display CMYK image
# ax3.set_title('CMYK')
# ax3.imshow(np.array(cmyk_image))
# ax3.axis('off')

# plt.show()

