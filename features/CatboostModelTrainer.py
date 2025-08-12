import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import os
from sklearn.feature_selection import SelectFromModel
import mlflow


class TrainCatboostModel():
    FEATURE_COLUMNS = ['MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC',
                  'z_score_0', 'z_score_1', 'z_score_2', 'z_score_3', 'z_score_4', 'z_score_5',
                  'max_abs_zscore', 'max_residual_zscore', 'max_consumption', 'min_consumption', 'consumption_range',
                  'electronic_weight_normalized', 'electromechanical_weight_normalized','common_weight_normalized', 'LABEL']

    CAT_FEATURES = ['MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC']

    NUM_FEATURES = ['z_score_0', 'z_score_1', 'z_score_2', 'z_score_3', 'z_score_4', 'z_score_5',
                    'max_abs_zscore', 'max_residual_zscore', 'max_consumption', 'min_consumption', 'consumption_range',
                    'electronic_weight_normalized', 'electromechanical_weight_normalized','common_weight_normalized']                  

    COLUMNS_DTYPE= {'CONS_ID': str, 'CSQ': str, 'CNO': str, 'MNO': str, 'MCD': str, 'AMP': str, 'VLT': str}

    TARGET_COL = 'LABEL'

    def __init__(self, train_df, output_directory):
        # train_df = pd.read_csv(data_file, usecols = TrainCatboostModel.FEATURE_COLUMNS, dtype = TrainCatboostModel.COLUMNS_DTYPE) # type: ignore

        X = train_df.drop(TrainCatboostModel.TARGET_COL, axis=1)
        y = train_df[TrainCatboostModel.TARGET_COL]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        cw = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train).tolist()
        class_weights = {0: cw[0], 1: cw[1]}

        self.catboost_model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.05,
            depth=6,
            class_weights=class_weights,
            eval_metric='F1',
            cat_features=TrainCatboostModel.CAT_FEATURES,
            early_stopping_rounds=100,
            verbose=100,
            bagging_temperature=0.5,
            border_count=64,
            l2_leaf_reg = 1
        )
        self.model_output_directory = output_directory

    def train_and_save_model(self):
        self.catboost_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=100,
            verbose=100
            )
        
        save_dir = os.path.join(self.model_output_directory, "catboost_models")
        os.makedirs(save_dir, exist_ok=True)
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., 20250725_134520
        model_filename = f"catboost_model_{timestamp}.cbm"

        # Full path
        model_path = os.path.join(save_dir, model_filename)

        # Save the model
        self.catboost_model.save_model(model_path)

        print(f"Model saved at: {model_path}")
        return self.catboost_model

    def feature_selection_pipeline(self):
        """Pipeline with CatBoost-based feature selection."""

        cw = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train).tolist()
        class_weights = {0: cw[0], 1: cw[1]}
        
        categorical_indices = [self.X_train.columns.get_loc(col) for col in TrainCatboostModel.CAT_FEATURES]
        original_dataset = mlflow.data.from_pandas(self.X_train, name="training_with_all_features")
        
        with mlflow.start_run(run_name="Feature Selection Pipeline") as run:
            mlflow.log_input(original_dataset, context="training_with_all_features")
            # Step 1: Train initial model for feature selection
            selector_model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.05,
            depth=6,
            class_weights=class_weights,
            eval_metric='F1',
            cat_features=TrainCatboostModel.CAT_FEATURES,
            early_stopping_rounds=100,
            verbose=100,
            bagging_temperature=0.5,
            border_count=64,
            l2_leaf_reg = 1
        )
            selector_model.fit(self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    early_stopping_rounds=100,
                    verbose=100)

            # Step 2: Feature selection based on importance
            selector = SelectFromModel(
                selector_model,
                threshold="median",  # Select features above median importance
                prefit=True,
            )

            X_train_selected = selector.transform(self.X_train)
            X_test_selected = selector.transform(self.X_val)
            
            mask = np.array([True, True, False, False, False, False, True, True, False, True, True, True, False, True, False, True, False, False, True, False])

            # Restore the categorical feature indices in the selected features
            selected_categorical_indices = [i for i in categorical_indices if mask[i]]
            # Convert to CSR and proceed with CatBoost
            print("Selected categorical indices:", selected_categorical_indices)
 
            column_names = ['MFG', 'AMP', 'z_score_0', 'z_score_1', 'z_score_3', 'z_score_4', 'z_score_5', 'max_residual_zscore', 'min_consumption', 'electromechanical_weight_normalized']
            #Before encoding
            print("Training data shape before encoding:", X_train_selected.shape)
            print("Test data shape before encoding:", X_test_selected.shape)
            
            print("Original Training data shape:", self.X_train.shape)
            print("Original Test data shape:", self.X_val.shape)
            print("Training data shape:", X_train_selected.shape)
            print("Test data shape:", X_test_selected.shape)
            print("Original training columns: ======= ", self.X_train.shape[1])
            print("Original test columns:", self.X_val.shape[1])
            print("Selected training columns:", X_train_selected.shape[1])
            print("Selected test columns:", X_test_selected.shape[1])
            # Log feature selection results
            selected_features = selector.get_support()
            n_selected = sum(selected_features)
            feature_names = self.X_train.columns.tolist()
            print("Original features =================", feature_names)
            print("Selected features =================== ", selected_features)
            
            mlflow.log_metrics(
                {
                    "original_features": self.X_train.shape[1],
                    "selected_features": n_selected,
                    "feature_reduction_ratio": n_selected / self.X_train.shape[1],
                }
            )

           
            # Step 3: Train final model on selected features
            SELECTED_CAT_FEATURES = ['MFG', 'AMP']
            final_model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.05,
            depth=6,
            class_weights=class_weights,
            eval_metric='F1',
            cat_features=selected_categorical_indices,
            early_stopping_rounds=100,
            verbose=100,
            bagging_temperature=0.5,
            border_count=64,
            l2_leaf_reg = 1
            )
            feature_selected_dataset = mlflow.data.from_numpy(X_train_selected, name="training_with_selected_features")
            
            final_model.fit(X_train_selected, self.y_train,
                    eval_set=[(X_test_selected, self.y_val)],
                    early_stopping_rounds=100,
                    verbose=100)

            mlflow.log_input(feature_selected_dataset, context="training_with_fewer_features")
            # Evaluate performance
            original_train_score = selector_model.score(self.X_train, self.y_train)
            original_test_score = selector_model.score(self.X_val, self.y_val)
            train_score = final_model.score(X_train_selected, self.y_train)
            test_score = final_model.score(X_test_selected, self.y_val)

            mlflow.log_metrics(
                    {
                        "train_accuracy_original": original_train_score,
                        "test_accuracy_original": original_test_score,
                        "train_accuracy_selected": train_score,
                        "test_accuracy_selected": test_score,
                    }
                )
            # Log the final model and selector
            mlflow.sklearn.log_model(final_model, name="final_model")
            mlflow.sklearn.log_model(selector, name="feature_selector")
            
            mlflow.register_model(
            "runs:/{}/final_model".format(run.info.run_id), 
            "final_model"
            )
             
            mlflow.register_model(
            "runs:/{}/feature_selector".format(run.info.run_id), 
            "feature_selector"
            )
            
            return final_model, selector