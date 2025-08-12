import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from datetime import datetime


class PrepareModelInputData:
    # ---- Rule Weights ----
    ELECTRONIC_WEIGHTS = {'t_1_c_1': 0.35, 't_1_c_2': 0.25, 't_1_c_3': 0.39}
    ELECTROMECHANICAL_WEIGHTS = {'t_2_c_1': 0.45, 't_2_c_2': 0.55, 't_2_c_3': 0.3}
    COMMON_WEIGHTS = {'t_a_c_1': 0.32, 'consecutive_NR': 0.35}
    NORMALISE_WEIGHTS_STATS = {
    'electronic_weight': {'mean': None, 'std': None},
    'electromechanical_weight': {'mean':  None, 'std':  None},
    'common_weight': {'mean': None,   'std': None}   
    }
    PREDICTION_STATS = {
    'electronic_weight': {'mean': 0.0235, 'std':0.1051},
    'electromechanical_weight': {'mean':  0.0352, 'std':  0.1106},
    'common_weight': {'mean': 0.0269,   'std': 0.1001}   
    }


    def __init__(self, df: pd.DataFrame, is_training_data = False):
        self.df = df.copy()
        self.base_columns = [
            'CONS_ID', 'CSQ', 'CNO', 'MNO', 'MCD', 'MFG', 'AMP', 'VLT',
            'SUP', 'SMT', 'ELC', 'defect_range', 'LABEL'
        ]
        self.zscore_columns = [
            'z_score_0', 'z_score_1', 'z_score_2', 'z_score_3',
            'z_score_4', 'z_score_5', 'max_abs_zscore',
            'max_residual_zscore', 'max_consumption',
            'min_consumption', 'consumption_range'
        ]
        if(is_training_data):
          self.NORMALISE_WEIGHTS_STATS = self.PREDICTION_STATS


    def prepare_data(self):
        t_df = self.df[self.base_columns].copy()

        # ---- Process consumption stats ----
        self.df['adv_ut_list'] = self.df['adv_ut_list'].apply(lambda x: pd.Series(x).fillna(0).to_numpy())
        results = np.array([self.process_consumption(cons) for cons in self.df['adv_ut_list']])
        t_df[self.zscore_columns] = np.vstack(results)

        for col in ['min_consumption', 'max_consumption', 'consumption_range']:
            t_df[col] = np.log1p(t_df[col])

        # ---- Apply binary rule functions ----
        for name, func in self.rule_functions().items():
            t_df[name] = self.df.apply(func, axis=1)

        # ---- Weighted score columns ----
        t_df['electronic_weight'] = t_df.apply(lambda row: self.compute_weight(row, self.ELECTRONIC_WEIGHTS), axis=1)
        t_df['electromechanical_weight'] = t_df.apply(lambda row: self.compute_weight(row, self.ELECTROMECHANICAL_WEIGHTS), axis=1)
        t_df['common_weight'] = t_df.apply(lambda row: self.compute_weight(row, self.COMMON_WEIGHTS), axis=1)

        # ---- Normalize weights ----
        t_df = self.normalize_weight_columns(t_df, self.NORMALISE_WEIGHTS_STATS)

        return t_df

    # ---- Rule Functions Map ----
    @staticmethod
    def rule_functions():
        return {
            't_1_c_1': PrepareModelInputData.check_t_1_c_1,
            't_1_c_2': PrepareModelInputData.check_t_1_c_2,
            't_1_c_3': PrepareModelInputData.check_t_1_c_3,
            't_2_c_1': PrepareModelInputData.check_t_2_c_1,
            't_2_c_2': PrepareModelInputData.check_t_2_c_2,
            't_2_c_3': PrepareModelInputData.check_t_2_c_3,
            't_a_c_1': PrepareModelInputData.check_t_a_c_1,
            'consecutive_NR': PrepareModelInputData.check_consecutive_NR
        }

    # ---- Rule Definitions ----
    @staticmethod
    def check_t_1_c_1(row):
        return int(row['ELC'] == 'N' and 'SP' in row['ent_cd_list'][-3:])

    @staticmethod
    def check_t_1_c_2(row):
        return int(row['ELC'] == 'N' and row['adv_ut_list'][-1] < 10)

    @staticmethod
    def check_t_1_c_3(row):
        return int(row['ELC'] == 'N' and 'LW' in row['ent_cd_list'][-3:])

    @staticmethod
    def check_t_2_c_1(row):
        if row['ELC'] == 'Y' and len(row['adv_ut_list']) >= 7:
            last = row['adv_ut_list'][-1]
            avg = np.mean(row['adv_ut_list'][-7:-1])
            return int(avg > 50 and last > 4 * avg)
        return 0

    @staticmethod
    def check_t_2_c_2(row):
        if row['ELC'] == 'Y' and len(row['adv_ut_list']) >= 7:
            last = row['adv_ut_list'][-1]
            avg = np.mean(row['adv_ut_list'][-7:-1])
            return int(avg > 50 and last > 8 * avg)
        return 0

    @staticmethod
    def check_t_2_c_3(row):
        return int(row['ELC'] == 'Y' and 'NR' in row['ent_cd_list'][-3:])

    @staticmethod
    def check_t_a_c_1(row):
        return int(row['ent_cd_list'] and row['ent_cd_list'][-1] in {'LCC', 'NR', 'Read'})

    @staticmethod
    def check_consecutive_NR(row):
        lst = row['ent_cd_list']
        return int(any(a == b == 'NR' for a, b in zip(lst, lst[1:])))

    @staticmethod
    def safe_zscore(arr):
        std = np.std(arr, ddof=0)
        if std < 1e-8:
            return np.zeros_like(arr)  # or np.full_like(arr, np.nan), if you prefer
        return zscore(arr, ddof=0)

    # ---- Utility Functions ----
    def process_consumption(self, consumption):
        try:
            max_consumption = np.max(consumption)
            min_consumption = min((x for x in consumption if x > 0), default=0)
            consumption_range = max_consumption - min_consumption
        except Exception as e:
            print(f"Consumption error: {e}")
            return [0] * len(self.zscore_columns)

        try:
            z = PrepareModelInputData.safe_zscore(consumption)
            z_tail = [z[-(i + 1)] if len(z) > i else 0 for i in range(6)]
            max_abs_z = np.max(np.abs(z)) if len(z) > 0 else 0
        except Exception as e:
            print(f"Z-score error: {e}")
            z_tail = [0] * 6
            max_abs_z = 0

        max_residual_z = -99
        if np.count_nonzero(consumption) >= len(consumption) - 3 and len(consumption) >= 24:
            try:
                result = seasonal_decompose(consumption, model='additive', period=12, extrapolate_trend='freq')
                clean_resid = result.resid[~np.isnan(result.resid)]
                if len(clean_resid) > 0:
                    max_residual_z = np.max(np.abs(zscore(clean_resid)))
            except Exception as e:
                print(f"Decomposition error: {e}")

        return z_tail + [max_abs_z, max_residual_z, max_consumption, min_consumption, consumption_range]

    # ---- Weight Scoring ----
    @staticmethod
    def compute_weight(row, weight_dict):
        return sum(row[rule] * weight for rule, weight in weight_dict.items())


    @staticmethod
    def normalize_weight_columns(df: pd.DataFrame,
                                 stats: dict[str, dict[str, float]],
                                 k: float = 6.0) -> pd.DataFrame:
        for col, s in stats.items():
            μ = s.get("mean")
            σ = s.get("std")
            if μ is None or σ is None: 
                print('HERE')                     # compute only when missing
                μ = df[col].mean() if μ is None else μ
                σ = df[col].std(ddof=0) if σ is None else σ
                σ = σ + 1e-8                                      # numerical safety
            df[f"{col}_normalized"] = 1.0 / (1.0 + np.exp(-k * (df[col] - μ) / σ))

            # optional: quick log to keep the nice console output you had
            print(f"{col}_mean: {μ:.4f}, {col}_std: {σ:.4f}")
        return df


