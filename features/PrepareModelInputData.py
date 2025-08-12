import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from core import AppGlobalVariables as agv
from core import custom_rules

class PrepareModelInputData():

    rules_with_weights = [(getattr(custom_rules, fn_name), weight) for fn_name, weight in agv.RULES_WITH_WEIGHTS.items()]


    def __init__(self, prepare_prediction_data_df):
        self.prepare_prediction_data_df = prepare_prediction_data_df

    def compute_score(self, row):
        return sum(weight for fn, weight in PrepareModelInputData.rules_with_weights if fn(row))

    def prepare_data(self):
        t_df = pd.DataFrame()
        t_df = self.prepare_prediction_data_df[['CONS_ID', 'CSQ', 'CNO', 'MNO', 'MCD', 'MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC', 'defect_range', 'LABEL']].copy()

        t_df['has_consecutive_NR_in_ENT_CD'] = self.prepare_prediction_data_df['ent_cd_list'].apply(self.has_consecutive_NR)

        t_df['computed_score'] = self.prepare_prediction_data_df.apply(self.compute_score, axis=1)

        # t_df['adv_ut_list'] = self.prepare_prediction_data_df['adv_ut_list'].apply(lambda consumptions: np.nan_to_num(consumptions, nan=0))

        #t_df[['max_consumption', 'min_consumption', 'z_score_0', 'z_score_1', 'z_score_2', 'z_score_3', 'z_score_4', 'z_score_5', 'max_abs_zscore', 'max_residual_zscore', 'consumption_range']] = self.prepare_prediction_data_df['adv_ut_list'].apply(lambda x: pd.Series(self.process_consumption(x)))
  
        # Convert the 'adv_ut_list' column into a list of arrays
        consumptions = self.prepare_prediction_data_df['adv_ut_list'].to_numpy()

        # Process all rows at once using list comprehension (faster than apply)
        results = np.array([self.process_consumption(x) for x in consumptions])

        # Assign the results back to the DataFrame
        columns = ['z_score_0', 'z_score_1', 'z_score_2', 
                'z_score_3', 'z_score_4', 'z_score_5', 'max_abs_zscore', 'max_residual_zscore', 'max_consumption', 'min_consumption', 'consumption_range']
        t_df[columns] = np.vstack(results)

        t_df['min_consumption'] = np.log1p(t_df['min_consumption'])
        t_df['max_consumption'] = np.log1p(t_df['max_consumption'])
        t_df['consumption_range'] = np.log1p(t_df['consumption_range'])
        return t_df

    def has_consecutive_NR(self, lst):
        for i in range(len(lst) - 1):
            if lst[i] == 'NR' and lst[i + 1] == 'NR':
                return 1
        return 0

    def process_consumption(self, consumptions):
        consumption = np.nan_to_num(consumptions, nan=0)

        max_consumption = min_consumption = consumption_range = 0
        try:
            max_consumption = consumption.max()
            min_consumption = min((num for num in consumption if num != 0), default=0)
            consumption_range = max_consumption - min_consumption
        except Exception as e:
            print(f"Exception message: {e}")
            return None
        
        z_score_0 = z_score_1 = z_score_2 = z_score_3 = z_score_4 = z_score_5 = 0
        consumption_zscore = zscore(consumption)
        length_of_child_consumption_zscore = len(consumption_zscore)
        if length_of_child_consumption_zscore > 5:
            z_score_0 = consumption_zscore[length_of_child_consumption_zscore - 1]
            z_score_1 = consumption_zscore[length_of_child_consumption_zscore - 2]
            z_score_2 = consumption_zscore[length_of_child_consumption_zscore - 3]
            z_score_3 = consumption_zscore[length_of_child_consumption_zscore - 4]
            z_score_4 = consumption_zscore[length_of_child_consumption_zscore - 5]
            z_score_5 = consumption_zscore[length_of_child_consumption_zscore - 6]
        max_abs_zscore = max(abs(consumption_zscore))

        num_zeros = len(consumption) - np.count_nonzero(consumption)
        if num_zeros < 3 and len(consumption) >= 24:
            result = seasonal_decompose(consumption, model='additive', period=12)
            residuals = result.resid
            mask = ~np.isnan(residuals)
            clean_residuals = residuals[mask]
            residuals_zscore = zscore(clean_residuals)
            residuals_zscore[np.isnan(residuals_zscore)] = 0
            max_residual_zscore = max(abs(residuals_zscore))
        else:
            max_residual_zscore = -99
        
        return z_score_0, z_score_1, z_score_2, z_score_3, z_score_4, z_score_5, max_abs_zscore, max_residual_zscore, max_consumption, min_consumption, consumption_range


