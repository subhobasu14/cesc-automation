import pandas as pd

class MeterDefectRange():
    def __init__(self, defect_range_file_path, mcd_file_path):
        self.defective_meter_dict  = self.get_meter_defect_range_dict(defect_range_file_path)  
        self.mcd_dict = self.get_mcd_dict(mcd_file_path)
        self.mcd_to_properties_dict = self.get_meter_properties_from_mcd(mcd_file_path)

    def get_meter_properties_from_mcd(self, mcd_file_path):
        mcd_df = pd.read_csv(mcd_file_path, usecols=['MCD', 'MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC'], dtype={'MCD': int})
        mcd_to_properties_dict = mcd_df.set_index('MCD')[['MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC']].to_dict(orient='index')
        return mcd_to_properties_dict
        
    def get_meter_defect_range_dict(self, defect_range_file):
        defective_meters_df = pd.read_csv(defect_range_file)
        defective_meter_dict = self.create_defective_meter_numbers(defective_meters_df)
        return defective_meter_dict
    
    def get_mcd_dict(self, mcd_file_path):
        mcd_df = pd.read_csv(mcd_file_path, usecols=['MCD', 'MFG', 'AMP', 'VLT'], dtype={'MCD': int})
        mcd_dict = mcd_df.set_index('MCD').apply(lambda row: f"{row['MFG']}|{row['AMP']}|{row['VLT']}", axis=1).to_dict()
        return mcd_dict

    def createCompositeKey(self, mfg, current_rating, volt_rating):
        dict_key = str(mfg) + '|' + str(current_rating) + '|' + str(volt_rating)
        return dict_key

    def create_defective_meter_numbers(self, df):
        defective_meter_data = {}
        for index, row in df.iterrows():
            mfg = row['MANUFACTURER']
            current_rating = row['CURRENT RATING']
            volt_rating = row['VOLTAGE RATING']
            start_sl_no = row['STARTING SERIAL NO.']
            end_sl_no = row['END SERIAL NO.']
            dict_key = self.createCompositeKey(mfg, current_rating, volt_rating)
            if dict_key not in defective_meter_data:
                defective_meter_data[dict_key] = set()
            for meter_no in range(start_sl_no, end_sl_no + 1):
                defective_meter_data[dict_key].add(meter_no)
        return defective_meter_data
    
    def check_defect_range(self, row):
        if row['mcd_value'] in self.defective_meter_dict:  # Check if `mcd_value` exists as a key
            if int(row['MNO']) in self.defective_meter_dict[row['mcd_value']]:  # Check if `MNO` is in the values
                return 1
        return 0
