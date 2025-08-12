import pandas as pd
import traceback 
from datetime import datetime
import numpy as np

from core import AppGlobalVariables as agv
from core.Helper import Helper



class OrchestratorForTraining():
    def __init__(self, combined_df):
        self.combined_df = combined_df
        self.outlier_df = pd.DataFrame()

    def get_outliers(self):
        return self.outlier_df[['CONS_ID', 'CSQ', 'CNO', 'MNO', 'REMARKS']].reset_index(drop=True)
    
    def manage_outliers(self, df, reason):
        df = df.copy()
        #df.loc[:, agv.rejection_reason_column_name] = reason
        df[agv.rejection_reason_column_name] = reason
        self.outlier_df = pd.concat([self.outlier_df, df], ignore_index=True)

    def merge_prediction_and_outliers(self, predictions_df):
        all_rows_df = pd.concat([predictions_df, self.get_outliers()], axis=0)
        return all_rows_df
        
    
    def get_combined_raw_data(self):
        try:
            start_time = datetime.now()
            master_with_consumptions_df = self.combined_df
            end_time = datetime.now()
            print(f"Time taken to extract exact consumption: {end_time - start_time}")
            start_time = datetime.now()
            master_with_consumptions_df['adv_ut_list'] = master_with_consumptions_df.apply(
                lambda row: Helper.process_bi_tri(row['CNO'], row['adv_ut_list']) if row['CNO'].startswith(('88', '011')) else row['adv_ut_list'],
                axis=1
            )
            end_time = datetime.now()
            print(f"Time taken to implement bi tri: {end_time - start_time}")
            master_with_consumptions_df['adv_ut_list'] = master_with_consumptions_df['adv_ut_list'].apply(lambda lst: lst[-36:])
            master_with_consumptions_df['ent_cd_list'] = master_with_consumptions_df['ent_cd_list'].apply(lambda lst: lst[-36:])

            master_with_consumptions_df['list_length'] = master_with_consumptions_df['adv_ut_list'].apply(len)
            consumer_with_less_than_6_consumption_df = master_with_consumptions_df[master_with_consumptions_df['list_length'] <= agv.consumption_min_count]
            self.manage_outliers(consumer_with_less_than_6_consumption_df, agv.row_filter_errors["less_consumption"])
            master_with_consumptions_df = master_with_consumptions_df[master_with_consumptions_df['list_length'] > agv.consumption_min_count]
            master_with_consumptions_df = master_with_consumptions_df.reset_index(drop=True)
            print('after max 4 consumption filter:', master_with_consumptions_df.shape, '\n', master_with_consumptions_df['is_current_meter'].value_counts())
            
            master_with_consumptions_df['max_value'] = master_with_consumptions_df['adv_ut_list'].apply(
                lambda x: np.nan if not isinstance(x, list) or all(pd.isna(x)) else np.nanmax(x)
            )
            master_with_consumptions_df = master_with_consumptions_df[master_with_consumptions_df['max_value'] > 1].copy()
            master_with_consumptions_df = master_with_consumptions_df.reset_index(drop=True)
            print('after min consumption > 1 filter:', master_with_consumptions_df.shape, '\n', master_with_consumptions_df['is_current_meter'].value_counts())

            master_with_consumptions_df['adv_ut_list_length'] = master_with_consumptions_df['adv_ut_list'].apply(lambda x: len(x) if isinstance(x, list) else np.nan)
            master_with_consumptions_df['adv_ut_nan_count'] = master_with_consumptions_df['adv_ut_list'].apply(lambda x: sum(item is np.nan for item in x) if isinstance(x, list) else np.nan)
            master_with_consumptions_df['nan_to_length_ratio'] = master_with_consumptions_df['adv_ut_nan_count'] / master_with_consumptions_df['adv_ut_list_length']
            master_with_consumptions_df = master_with_consumptions_df[master_with_consumptions_df['nan_to_length_ratio'] <= agv.consumption_nan_to_val_cutoff_ratio].copy()
            master_with_consumptions_df = master_with_consumptions_df.reset_index(drop=True)
            print('after nan to length cut off filter:', master_with_consumptions_df.shape, '\n', master_with_consumptions_df['is_current_meter'].value_counts())

            start_time = datetime.now()
            master_with_consumptions_df['mcd_value'] = master_with_consumptions_df['MCD'].apply(lambda x: agv.meter_defect_range.mcd_dict.get(x, None))
            master_with_consumptions_df['defect_range'] = master_with_consumptions_df.apply(agv.meter_defect_range.check_defect_range, axis=1)
            defect_range_meters_df = master_with_consumptions_df[master_with_consumptions_df['defect_range']==1]
            meters_for_prediction_df = master_with_consumptions_df[master_with_consumptions_df['defect_range']==0]
            print('after defect range filter:', meters_for_prediction_df.shape, '\n', meters_for_prediction_df['is_current_meter'].value_counts())
            self.manage_outliers(defect_range_meters_df, agv.row_filter_errors["defect_range"])
            end_time = datetime.now()
            print(f"Time taken to implement defect range: {end_time - start_time}")

            start_time = datetime.now()
            not_supported_mcd_df = meters_for_prediction_df[~meters_for_prediction_df['MCD'].isin(agv.supported_mcd_values)]
            self.manage_outliers(not_supported_mcd_df, agv.row_filter_errors["nonsupported_mcd_values"])
            meters_for_prediction_df = meters_for_prediction_df[meters_for_prediction_df['MCD'].isin(agv.supported_mcd_values)]
            meters_for_prediction_df = meters_for_prediction_df.reset_index(drop=True)
            not_supported_mcd_df.to_csv('not_supported_mcd_df.csv', index=False)
            print('after supported mcd filter:', meters_for_prediction_df.shape, '\n', meters_for_prediction_df['is_current_meter'].value_counts())
            end_time = datetime.now()
            print(f"Time taken to implement MCD filter: {end_time - start_time}")

            start_time = datetime.now()
            meters_for_prediction_df[['MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC']] = meters_for_prediction_df['MCD'].apply(
                lambda x: pd.Series({
                    'MFG': agv.meter_defect_range.mcd_to_properties_dict.get(x, {}).get('MFG'),
                    'AMP': agv.meter_defect_range.mcd_to_properties_dict.get(x, {}).get('AMP'),
                    'VLT': agv.meter_defect_range.mcd_to_properties_dict.get(x, {}).get('VLT'),
                    'SUP': agv.meter_defect_range.mcd_to_properties_dict.get(x, {}).get('SUP'),
                    'SMT': agv.meter_defect_range.mcd_to_properties_dict.get(x, {}).get('SMT'),
                    'ELC': agv.meter_defect_range.mcd_to_properties_dict.get(x, {}).get('ELC')
                })
            )
            end_time = datetime.now()
            print(f"Time taken to add MCD properties: {end_time - start_time}")

            meters_for_prediction_df['LABEL'] = meters_for_prediction_df['is_current_meter'].map({True: 0, False: 1})
            print('final data:', meters_for_prediction_df.shape, '\n', meters_for_prediction_df['LABEL'].value_counts())
            
            return meters_for_prediction_df            
        except:
            traceback.print_exc()  
""" 
    def get_prediction_data(self, meters_for_prediction_df):
        try: 
            start_time = datetime.now()
            prepare_model_input_data = PrepareModelInputData(meters_for_prediction_df)
            model_input_df = prepare_model_input_data.prepare_data()
            end_time = datetime.now()
            print(f"Time taken to prepare model input data: {end_time - start_time}")
            # print('model input column:', model_input_df.columns[5:35])
            # start_time = datetime.now()
            # make_predictions = MakePrediction(model_input_df)
            # prediction_data_df = make_predictions.generate()
            # end_time = datetime.now()
            # print(f"Time taken to make predictions: {end_time - start_time}")
            # final_df = self.merge_prediction_and_outliers(prediction_data_df)
            return model_input_df
        except:
            traceback.print_exc() 

        """