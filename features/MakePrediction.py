import pandas as pd
from catboost import CatBoostClassifier
from core import AppGlobalVariables as agv

class MakePrediction:
    model = None  # Placeholder for the model
    
    @classmethod
    def load_model(cls):
        if cls.model is None:
            cls.model = CatBoostClassifier().load_model(agv.model_path)

    def __init__(self, prediction_data_df: pd.DataFrame):
        self.load_model()  # Ensure the model is loaded
        self.prediction_data_df = prediction_data_df.reset_index(drop=True)
        self.prediction_data_df.to_csv('data_for_prediction.csv', index=False)
        self.identifiers = self.prediction_data_df[['CONS_ID', 'CSQ', 'CNO', 'MNO']]
        print("self.identifiers", self.identifiers.shape)

    def generate(self) -> pd.DataFrame:
        """Generates predictions and returns a DataFrame with prediction scores and categories."""
        X_test = self.prediction_data_df.drop(columns=['CONS_ID', 'CSQ', 'CNO', 'MNO'])
        
        # Predict probabilities
        probabilities = MakePrediction.model.predict_proba(X_test)
        
        # Construct DataFrame efficiently
        proba_df = self.identifiers.assign(
            PREDICTION_SCORE=probabilities[:, 1],  # Extract class_1 probability directly
            CATEGORY=pd.cut(probabilities[:, 1], bins=agv.prediction_bins, labels=agv.prediction_category, right=False)
        )
        print("proba_df",proba_df.shape)

        all_rows_with_prediction_df = pd.concat([self.prediction_data_df, proba_df], axis=1)
        all_rows_with_prediction_df.to_csv('training_data_with_prediction.csv', index=False)
        return proba_df
