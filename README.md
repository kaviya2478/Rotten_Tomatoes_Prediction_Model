Movie Audience Rating Prediction
This project predicts the audience rating of movies based on various features like genre, cast, director, studio, and more using machine learning algorithms. 
The model used for prediction is a Random Forest Regressor, and the project involves several stages, including data loading, preprocessing, model training, hyperparameter tuning, and saving/loading the model for predictions.
Here’s a breakdown of each step in the pipeline:
  1. Loading Data:
   - Function: `load_data(file_path)`
   - Purpose: The `load_data()` function reads the dataset from an Excel file (`Rotten_Tomatoes_Movies.xls`) and prints the information about the DataFrame, including column names, data types, and the number of non-null entries.
   
  2. Preprocessing the Data:
     - Function: `preprocess_data(df)`
     - Purpose: Prepares the data for training the model by performing several tasks:
     - Dropping Unnecessary Columns: Removes columns that are not relevant for prediction (`movie_title`, `movie_info`, `critics_consensus`).
     - Handling Missing Values: Fills missing values in the `runtime_in_minutes` column with the median value. Missing values in the `studio_name` column are filled with `'Unknown'`. Rows with missing `audience_rating` values are dropped.
     - Datetime Conversion: Converts `in_theaters_date` and `on_streaming_date` columns to numeric values by calculating the difference in days from a reference date (2020-01-01).
     - Encoding Categorical Variables: Converts categorical columns (`rating`, `genre`, `directors`, `writers`, `cast`, `studio_name`, `tomatometer_status`) into numerical values using `LabelEncoder`. A separate encoder is used for each column, and the encoders are stored in a dictionary.
     - Defining Features and Target Variables: The features (`X`) are the columns excluding `audience_rating`, which is the target variable (`y`).

 3. Train-Test Split:
   - Function: `train_test_split_data(X, y)`
   - Purpose: Splits the data into training and testing sets (80% for training and 20% for testing) using `train_test_split` from `sklearn.model_selection`.

 4. Training the Random Forest Model:
     - Function: `train_random_forest(X_train, y_train, X_test, y_test)`
     - Purpose: Trains a Random Forest Regressor model on the training data (`X_train`, `y_train`).
     - The model is evaluated on the test data (`X_test`, `y_test`), and performance metrics (Root Mean Squared Error (RMSE) and R² score) are printed.
     - `n_estimators=100` sets the number of trees in the forest, and `max_depth=10` limits the depth of each tree.

5. Hyperparameter Tuning with Grid Search:
     - Function: `tune_hyperparameters(X_train, y_train)`
     - Purpose: Performs hyperparameter tuning to find the best parameters for the Random Forest model using `GridSearchCV`. The grid search explores different combinations of parameters like:
     - `n_estimators`: Number of trees in the forest.
     - `max_depth`: Maximum depth of the trees.
     - `min_samples_split`: Minimum number of samples required to split an internal node.
     - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
     - The best model (with the best hyperparameters) is returned.

6. Saving the Model:
   - Function: `save_model(model, file_name)`
   - Purpose:Saves the trained and optimized model to a file (`optimized_random_forest.pkl`) using `joblib.dump()`. This allows the model to be reused later without retraining.

7. Loading the Model:
   - Function: `load_model(file_name)`
   - Purpose: Loads the saved model from the file (`optimized_random_forest.pkl`) using `joblib.load()`. This model can now be used to make predictions on new data.

8. Prediction on New Data:
   - Example Input Data: The input for prediction consists of new data about a movie (represented as a DataFrame). The columns in this input data must match the columns the model was trained on, both in terms of feature order and the presence of all necessary columns.
   - Missing Columns: If any features are missing from the input data (e.g., a feature was not included in the example), the missing columns are added with a default value of 0 to match the training data.
   - Reordering Columns: The columns of the input data are reordered to match the order of features in the trained model (`trained_features`).
   - Prediction: The model predicts the audience rating for the input data using the `predict()` method, and the predicted value is printed.

Summary of Pipeline Flow:
1. Data Loading: Load movie dataset.
2. Preprocessing: Clean the data by dropping unnecessary columns, handling missing values, converting dates, and encoding categorical features.
3. Data Split: Split the data into training and testing sets.
4. Model Training: Train a Random Forest Regressor model on the training data.
5. Hyperparameter Tuning: Optimize the model's hyperparameters using GridSearchCV.
6. Model Saving: Save the trained and optimized model to a file.
7. Model Loading & Prediction: Load the model from the file and make predictions on new data.

This pipeline ensures that the data is properly cleaned, the model is optimized, and predictions can be made efficiently with the trained model.
