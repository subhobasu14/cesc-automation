import pandas as pd
from core import AppGlobalVariables as agv


class IdentifyConsumersForDataPreperation():
    
    def __init__(self, consumer_df, consumption_df):
        self.outlier_df = pd.DataFrame()
        self.master_df = consumer_df
        self.consumption_df = consumption_df

    def manage_outliers(self, df, reason):
        df[agv.rejection_reason_column_name] = reason
        self.outlier_df = pd.concat([self.outlier_df, df], ignore_index=True)

    def identify_commons_master_consumption(self):
        master_set = set(zip(self.master_df['CONS_ID'], self.master_df['CSQ']))
        consumption_set = set(zip(self.consumption_df['CONS_ID'], self.consumption_df['CSQ']))
        intersection_m_c = master_set.intersection(consumption_set)
        common_df = pd.DataFrame(intersection_m_c, columns=['CONS_ID', 'CSQ'])
        missing_consumptions_set = master_set - consumption_set
        missing_consumptions_df = pd.DataFrame(missing_consumptions_set, columns=['CONS_ID', 'CSQ'])
        return common_df, missing_consumptions_df

    def process_master(self, m_df):
        ma_df = m_df.drop_duplicates().copy()
        ma_df['FDT'] = pd.to_datetime(ma_df['FDT'], format='%d/%m/%Y', errors='coerce')
        master_missing_fdt_df = ma_df[ma_df['FDT'].isna()]
        # print("master_missing_fdt_df", master_missing_fdt_df.shape)
        self.manage_outliers(master_missing_fdt_df, agv.row_filter_errors["missing_fdt"])
        ma_df = ma_df[~ma_df['FDT'].isna()]
        ma_df['m_FDT'] = ma_df['FDT'].dt.to_period('M')
        if {'XCHG', 'ADV'}.issubset(ma_df.columns):
            ma_df['LABEL'] = ma_df.apply(lambda row: 1 if row['XCHG'] == 'Y' or row['ADV'] == 'Y' else 0, axis=1)
        return ma_df

    def process_consumption(self, c_df):
        c_df = c_df.drop_duplicates().copy()
        c_df['BC_CD'] = pd.to_datetime(c_df['BC_CD'].astype(str), format='%Y%m', errors='coerce').dt.to_period('M')
        return c_df

    def consumer_with_different_meter_in_m_c(self, m_df, c_df):
        latest_mno_m = m_df.sort_values(['CONS_ID', 'CSQ', 'FDT']).drop_duplicates(subset=['CONS_ID', 'CSQ'], keep='last')
        latest_mno_c = c_df.sort_values(['CONS_ID', 'CSQ', 'BC_CD']).drop_duplicates(subset=['CONS_ID', 'CSQ'], keep='last')
        compare_mno_df = latest_mno_m[['CONS_ID', 'CSQ', 'MNO']].merge(
            latest_mno_c[['CONS_ID', 'CSQ', 'MNO']], 
            on=['CONS_ID', 'CSQ'], 
            suffixes=('_m', '_c'), 
            how='inner'
        )

        different_meters = compare_mno_df.query("MNO_m != MNO_c")[['CONS_ID', 'CSQ']]
        self.manage_outliers(different_meters, agv.row_filter_errors["different_mno"])
        different_indices = set(zip(different_meters['CONS_ID'], different_meters['CSQ']))

        f_m_df = latest_mno_m[~latest_mno_m.set_index(['CONS_ID', 'CSQ']).index.isin(different_indices)].reset_index(drop=True)
        f_c_df = c_df[~c_df.set_index(['CONS_ID', 'CSQ']).index.isin(different_indices)].reset_index(drop=True)
        return f_m_df, f_c_df

    def get_valid_data(self):
        common_df, missing_consumptions_df = self.identify_commons_master_consumption()

        m_df = common_df.merge(self.master_df, on=['CONS_ID', 'CSQ'])
        c_df = common_df.merge(self.consumption_df, on=['CONS_ID', 'CSQ'])

        missing_consumptions_outlier_df = missing_consumptions_df.merge(self.master_df, on=['CONS_ID', 'CSQ'])
        # print("missing_consumptions_outlier_df", missing_consumptions_outlier_df.shape)

        self.manage_outliers(missing_consumptions_outlier_df, agv.row_filter_errors["no_consumption"])

        m_df = self.process_master(m_df)
        c_df = self.process_consumption(c_df)

        return self.consumer_with_different_meter_in_m_c(m_df, c_df)
        
    def get_outliers(self):
        return self.outlier_df[['CONS_ID', 'CSQ', 'CNO', 'MNO', 'REMARKS']].reset_index(drop=True)
