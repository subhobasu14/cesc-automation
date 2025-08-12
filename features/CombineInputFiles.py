import pandas as pd
import numpy as np
# import sys
# sys.path.append('/opt/airflow/core')
from core import AppGlobalVariables as agv
from features.Orchestrator import Orchestrator
from features.OrchestratorForTraining import OrchestratorForTraining
import ast

class CombineConsumerAndConsumption():
    
    def __init__(self, consumer_file, consumption_file, report_file = None):
        self.consumers_df = pd.read_csv(consumer_file,
                                        encoding='utf-8',
                                        sep=agv.csv_file_delimeter,
                                        dtype=agv.consumers_file_column_dtype, # type: ignore
                                        usecols=agv.consumer_columns
                                        )

        self.consumptions_df = pd.read_csv(
                                        consumption_file,
                                        encoding='utf-8',
                                        sep=agv.csv_file_delimeter,
                                        dtype=agv.consumptions_file_column_dtype, # type: ignore
                                        usecols=agv.consumption_columns
                                        )
        self.report_df = pd.DataFrame()
        if (report_file != None):
            self.report_df = pd.read_excel(report_file, dtype = {'CONS_ID': str, 'CSQ': str, 'MNO': str, 'CNO': str}, usecols=['CONS_ID', 'CSQ', 'CNO', 'MNO', 'TD REMARKS'] )

        self.orchestrate = Orchestrator(self.consumers_df, self.consumptions_df)


    def get_combined_raw_data(self):
        raw_data_df = self.orchestrate.get_combined_raw_data()
        if not self.report_df.empty:
            raw_data_df = self.change_label_on_report(raw_data_df)
        return raw_data_df
    
    def change_label_on_report(self, df):
        df = pd.merge(df, self.report_df, on=['CONS_ID', 'CSQ', 'MNO', 'CNO'], suffixes=('_m', '_c'), how='inner')
        df = df[df['TD REMARKS'] != 'DIFFERENT METER AT SITE: METER OK'].reset_index(drop=True)
        df['U_LABEL'] = df.apply(self.assign_u_label, axis=1)
        df = df.drop(columns=['LABEL', 'TD REMARKS']).rename(columns={'U_LABEL': 'LABEL'})
        return df

    def assign_u_label(self,row):
        if row['LABEL'] == 1 and row['TD REMARKS'] == 'METER OK':
            return 0
        elif row['LABEL'] == 0 and row['TD REMARKS'] == 'METER DEFECTIVE':
            return 1
        else:
            return row['LABEL']
        
#    def get_prediction_data(self, df):
#        return self.orchestrate.get_prediction_data(df)
    
class CombineConsumerAndMeterHistory():

    def parse_list(self, s):
        return eval(s.replace("nan", "np.nan"))

    def __init__(self, master_file, _meter_life_consumption):
        master_df = pd.read_csv(master_file, sep='|', dtype = {'CONS_ID': str, 'CSQ': str, 'MNO': str, 'CONS_NUM': str})
        m_df = master_df[['CONS_ID', 'CSQ', 'CONS_NUM']].drop_duplicates().reset_index(drop=True)
        m_df = m_df.rename(columns={'CONS_NUM': 'CNO'})

        ml_org_df = pd.read_csv(_meter_life_consumption, dtype = {'CONS_ID': str, 'CSQ': str})
        ml_org_df['adv_ut_list'] = ml_org_df['adv_ut_list'].apply(self.parse_list)
        ml_org_df['bl_cd_list'] = ml_org_df['bl_cd_list'].apply(ast.literal_eval)
        ml_org_df['ent_cd_list'] = ml_org_df['ent_cd_list'].apply(ast.literal_eval)
        
        self.ml_c_df = pd.merge(ml_org_df, m_df, on=['CONS_ID', 'CSQ'], how='inner').sort_values(['CONS_ID', 'CSQ'])
        print('combined_data shape:', self.ml_c_df.shape)
        self.orchestrate = OrchestratorForTraining(self.ml_c_df)

    def get_combined_raw_data(self):
        raw_data_df = self.orchestrate.get_combined_raw_data()
        return raw_data_df
    
#    def get_prediction_data(self, df):
#        return self.orchestrate.get_prediction_data(df)
