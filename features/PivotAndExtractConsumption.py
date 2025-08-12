import pandas as pd
from datetime import datetime
import numpy as np

class PivotAndExtractConsumption():
    def __init__(self, master_df, consumption_df):
        self.outlier_df = pd.DataFrame()
        self.master_df = master_df.reset_index(drop=True)
        self.consumption_df = consumption_df.reset_index(drop=True)

    def orchestrate(self):
        start_time = datetime.now()
        m_meter_active_span_df = self.get_max_date_between_bccd_fdt()
        end_time = datetime.now()
        print(f"Time taken to get_max_date_between_bccd_fdt(): {end_time - start_time}")
        start_time = datetime.now()
        month_range_df = self.get_month_range(m_meter_active_span_df)
        end_time = datetime.now()
        print(f"Time taken to get_month_range(): {end_time - start_time}")
        start_time = datetime.now()
        adv_pivot_df, ent_ct_pivot_df = self.create_pivot(month_range_df)
        end_time = datetime.now()
        print(f"Time taken to create_pivot(): {end_time - start_time}")
        start_time = datetime.now()
        m_bc_cd_df = self.extract_consumption(m_meter_active_span_df, adv_pivot_df, ent_ct_pivot_df)
        end_time = datetime.now()
        print(f"Time taken to extract_consumption(): {end_time - start_time}")
        return m_bc_cd_df.reset_index(drop=True)

    def get_max_date_between_bccd_fdt(self):
        bc_cd_range_c_df = self.consumption_df.groupby(['CONS_ID', 'CSQ']).agg(
            min_BC_CD=('BC_CD', 'min'), 
            max_BC_CD=('BC_CD', 'max')
        )

        # Merge using index-based join
        self.master_df = self.master_df.set_index(['CONS_ID', 'CSQ'])
        m_bc_cd_df = self.master_df.join(bc_cd_range_c_df, how="left").reset_index()

        m_bc_cd_df['min_MNO_fix_date'] = m_bc_cd_df[['min_BC_CD', 'm_FDT']].max(axis=1)
        return m_bc_cd_df


    def get_month_range(self, m_bc_cd_df):
        start_date = m_bc_cd_df['min_MNO_fix_date'].min()
        end_date = m_bc_cd_df['max_BC_CD'].max()

        return pd.period_range(start=start_date, end=end_date, freq='M')


    def create_pivot(self, months_list_period):
        # Use .groupby() + .unstack() instead of pivot_table for performance gains
        ADV_UT_pivot_df = self.consumption_df.groupby(['CONS_ID', 'CSQ', 'BC_CD'])['ADV_UT'].sum().unstack(fill_value=np.nan)
        # ENT_CD_pivot_df = self.consumption_df.groupby(['CONS_ID', 'CSQ', 'BC_CD'])['ENT_CD'].first().unstack(fill_value=np.nan)

        # ADV_UT_pivot_df = self.consumption_df.groupby(['CONS_ID', 'CSQ', 'BC_CD'])['ADV_UT'].sum().unstack().applymap(lambda x: None if pd.isna(x) else x)
        ENT_CD_pivot_df = self.consumption_df.groupby(['CONS_ID', 'CSQ', 'BC_CD'])['ENT_CD'].first().unstack().applymap(lambda x: None if pd.isna(x) else x)

        # Reindex columns to match the desired period range
        ADV_UT_pivot_df = ADV_UT_pivot_df.reindex(columns=months_list_period, fill_value=np.nan)
        ENT_CD_pivot_df = ENT_CD_pivot_df.reindex(columns=months_list_period, fill_value=np.nan)

        # Ensure both have the same index
        all_index = ADV_UT_pivot_df.index.union(ENT_CD_pivot_df.index)
        ADV_UT_pivot_df = ADV_UT_pivot_df.reindex(index=all_index, fill_value=np.nan)
        ENT_CD_pivot_df = ENT_CD_pivot_df.reindex(index=all_index, fill_value=np.nan)

        return ADV_UT_pivot_df, ENT_CD_pivot_df

    def extract_consumption(self, m_bc_cd_df, ADV_UT_pivot_df, ENT_CD_pivot_df):
        # Create a mask to filter the relevant columns for each row
        min_dates = m_bc_cd_df['min_MNO_fix_date'].values[:, None]  # Convert to 2D array for broadcasting
        max_dates = m_bc_cd_df['max_BC_CD'].values[:, None]

        # Convert pivot table columns to NumPy array for fast lookup
        pivot_cols = np.array(ADV_UT_pivot_df.columns)

        # Create boolean mask: True if column falls within the range
        mask = (pivot_cols >= min_dates) & (pivot_cols <= max_dates)

        # Extract values using NumPy advanced indexing
        m_bc_cd_df['adv_ut_list'] = [
            row[mask[idx]].tolist() for idx, row in enumerate(ADV_UT_pivot_df.loc[m_bc_cd_df.set_index(['CONS_ID', 'CSQ']).index].values)
        ]

        m_bc_cd_df['ent_cd_list'] = [
            row[mask[idx]].tolist() for idx, row in enumerate(ENT_CD_pivot_df.loc[m_bc_cd_df.set_index(['CONS_ID', 'CSQ']).index].values)
        ]

        return m_bc_cd_df


