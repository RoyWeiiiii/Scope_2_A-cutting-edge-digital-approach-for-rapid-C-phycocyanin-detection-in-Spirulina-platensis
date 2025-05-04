# Artificial Intelligence (AI) approach for the quantification of C-phycocyanin in _Spirulina platensis_: Hybrid stacking-ensemble model based on machine learning and deep learning

The rising demand for natural pigments in nutraceuticals, pharmaceuticals, and cosmetics has highlighted the need for efficient, non-destructive methods to predict C-phycocyanin (CPC) concentrations in Spirulina platensis. Conventional extraction and quantification methods are labour-intensive, environmentally harmful, and time-consuming. This study proposes a hybrid stacking-ensemble model integrating convolutional neural networks (CNN) for automated feature extraction with both Support Vector Machine (SVM) and eXtreme gradient boosting (XGBoost) as base models and multiple meta-regressor models. The meta-regressors include Ridge regression with built-in cross-validation (RidgeCV), Linear Regression (LR), Support Vector Regressor (SVR), Decision Tree (DT), Random Forest (RF), and XGBoost. The datasets comprising 11,000 images of biomass and extracted CPC were captured under varying lighting conditions and device setups to reflect real-world conditions accurately. While digital cameras achieved higher accuracy, smartphones provide competitive results under both covered and light-disturbed conditions, demonstrating smartphones' scalability for real-time applications. While XGBoost as a meta-regressor (standard deviation (Std) = 0.0052, R2Train = 0.9985, R2Val = 0.9983 with 95% confident interval (CI) [0.9979, 0.9986]), demonstrates marginal improvements over individual models such as SVM (Std = 0.0065, R2Train = 0.9975, R2Val = 0.9973 with 95% CI [0.9970, 0.9976]) and XGBoost (Std= 0.0053, R2Train = 0.9987, R2Val = 0.9982 with 95% CI [0.9978, 0.9986]), yet ensemble approach offers lower variability, reduced overfitting, enhanced stability, and generalisation. CPC datasets delivered better accuracy but were competitive against biomass datasets. Thus, biomass datasets are more feasible in real-world applications by excluding the need for extraction steps, enabling rapid, reliable, and accurate CPC concentration predictions.

![Uploading Graphical_Abstract_V1.png…]()

**Keywords**: _Spirulina platensis_; C-phycocyanin; Deep learning (DL); Machine learning (ML); Ensemble learning (EL); Microalgae

# Folder & Files descriptions
# a) Machine learning, Deep learning, & Hybrid Stacking-ensemble models

**AI_models_Final** => Contains the model configuration of SVM regressor, XGBoost regressor, CNN, and Hybrid Stacking-Ensemble model [Base models (SVM, XGBoost) & meta-regressor models (RidgeCV, LinearRegression, DecisionTree, RandomForest, SVR, XGBoost)

**SVM-XGBoost-CNN-Hybrid-EL-Model_development.xlsx** => Development of all models and with detailed explanation on python code in excel file

**SVM-XGBoost-CNN-Hybrid-EL-Model-Datasets.xlsx** => Datasets with accuracy and loss metrics derived from each model in excel file

# b) Image and data pre-processing

**Colour feature extraction & Data normalisation** =>  Contains the python script of colour feature extraction & data normalisation for training of ML models

**SVM-XGBoost-Colour_feature_extraction_Data_normalisation.xlsx** => Colour (RGB, HSL, CMYK) feature extraction and Data normalisation (min-max scaler) for ML models in excel file

# c) Datasets for CNN & Hybrid Stacking-ensemble models

**Combined_All_Batch_Days_Camera** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using digital camera capturing device under various type of variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Combined_All_Batch_Days_Smartphone** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using smartphone capturing device under various type of colour variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

# d) Datasets for SVM & XGBoost models

**Data_Colour_index_Normalised (1V-Input)** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Day_Colour_index_Normalised (2V-Input)** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK) with additonal 'Day' (period), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

**Data_Abs_Day_Colour_index_Normalised (3V-Input)** => Contains the combined data for all batches (3) and days (2, 4, 6, 8, 10, & 12) when using various image capturing devices (digital camera/ smartphone), type of colour variables (colour models such as RGB, HSL, & CMYK) with additonal 'Day' (period),  'Abs' (absorbance), and lightning conditions (covered [not exposed to light]/ light disturbed [non_covered])

# e) _Spirulina platensis_ biomass & extracted CPC image dataset
The google drive [https://drive.google.com/drive/folders/1dXDUHCD9nTJaF0CFyxqUkHKZH8Ko1XjF?usp=drive_link] contains the cropped image dataset of _Spirulina platensis_ biomass & extracted CPC grown under BG-11 medium in the period of 12 days. Subsequently, each day will contain a subfolder of both image capturing devices such as smartphone [Model => Iphone_13_Pro_Max] and digital camera [Model => Nikon_Z50]. Each image capturing device will contain a subfolder of covered [images taken without any light disturbances] and light_disturbed [images taken under light disturbed condition]. The experiment is conducted for 3 batches. The image dataset is publicly available for academic and research purposes.

# Referencing and citation
If you find the prediction and analysis of C-phycocyanin (CPC) concentration as well as the image dataset useful in your research, please consider citing: Based on the DOI: *********Not published yet***********
