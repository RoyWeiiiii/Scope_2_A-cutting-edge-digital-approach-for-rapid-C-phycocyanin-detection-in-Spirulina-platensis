# Revolutionising biotechnology: A cutting-edge digital approach for rapid C-phycocyanin detection in _Spirulina platensis_

To be filled later on

**Keywords**: 

# Folder and files description

**AI_models_Final** => Contains the model configuration of SVM regressor, XGBoost regressor, CNN, and Hybrid Stacking-Ensemble model [Base models (SVM, XGBoost) & meta-regressor models (RidgeCV, LinearRegression, DecisionTree, RandomForest, SVR, XGBoost)

**Colour feature extraction & Data normalisation** =>  Contains the python script of colour feature extraction & data normalisation for training of ML models

**Combined_All_Batch_Days_Camera** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using digital camera capturing device under various type of variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Combined_All_Batch_Days_Smartphone** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using smartphone capturing device under various type of colour variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Abs_Day_Colour_index_Normalised** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK) with additonal 'Day' (period),  'Abs' (absorbance), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Colour_index_Normalised** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Day_Colour_index_Normalised** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK) with additonal 'Day' (period), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**SVM-XGBoost-CNN-Hybrid-EL-Model_development.xlsx** => Development of all models and with detailed explanation on python code in excel file

**SVM-XGBoost-CNN-Hybrid-EL-Model-Datasets.xlsx** => Datasets with accuracy and loss metrics derived from each model in excel file

**SVM-XGBoost-Colour_feature_extraction_Data_normalisation.xlsx** => Colour (RGB, HSL, CMYK) feature extraction and Data normalisation (min-max scaler) for ML models in excel file 

# _Spirulina platensis_ biomass & extracted CPC image dataset
The google drive [https://drive.google.com/drive/folders/1dXDUHCD9nTJaF0CFyxqUkHKZH8Ko1XjF?usp=drive_link] contains the cropped image dataset of _Spirulina platensis_ biomass & extracted CPC grown under BG-11 medium in the period of 12 days. Subsequently, each day will contain a subfolder of both image capturing devices such as smartphone [Model => Iphone_13_Pro_Max] and digital camera [Model => Nikon_Z50]. Each image capturing device will contain a subfolder of covered [images taken without any light disturbances] and light_disturbed [images taken under light disturbed condition]. The experiment is conducted for 3 batches. The image dataset is publicly available for academic and research purposes.

# Referencing and citation
If you find the prediction and analysis of C-phycocyanin (CPC) concentration as well as the image dataset useful in your research, please consider citing: Based on the DOI: *********Not published yet***********
