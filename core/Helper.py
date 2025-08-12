import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

class Helper():
    def __init__():
        pass

    @staticmethod  
    def process_bi(values):
        processed_values = []

        for i in range(len(values)):
            if pd.isna(values[i]):
                # Replace NaN with half of the next integer if available
                if i + 1 < len(values) and not pd.isna(values[i + 1]):
                    next_val = int(values[i + 1] / 2)
                    processed_values.append(next_val)
            else:
                # Halve the integer value
                processed_values.append(int(values[i] / 2))

        # Trim the last value if it's NaN
        if processed_values and pd.isna(processed_values[-1]):
            processed_values.pop()

        return processed_values
    
    @staticmethod
    def process_tri(nums):
        n = len(nums)
        i = 0

        # Traverse through the list
        while i < n:
            if not np.isnan(nums[i]):  # Check if the current value is a valid number
                val = int(nums[i] / 3)

                # Place the divided value in positions n-2, n-1, and n if within range
                if i >= 2:
                    nums[i-2] = val
                if i >= 1:
                    nums[i-1] = val
                nums[i] = val
            i += 1

        # Remove trailing np.NaN values from the list
        last_valid_idx = len(nums) - 1
        while last_valid_idx >= 0 and np.isnan(nums[last_valid_idx]):
            last_valid_idx -= 1

        return nums[:last_valid_idx + 1]
    
    @staticmethod
    def process_bi_tri(cno, num_list):
        if cno.startswith('011'):
            return Helper.process_bi(num_list)
        if cno.startswith('88'):
            return Helper.process_tri(num_list)
        else:
            return num_list

    @staticmethod   
    def generate_evaluation_metric(df, actual, pred):
        y_true = df[actual]
        y_pred = df[pred]

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)

        # Classification report (includes all the above per class)
        print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()