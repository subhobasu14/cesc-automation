import pandas as pd

class FileConcatenator():
    def __init__(self, file_priority_dict):
        self.file_dict = file_priority_dict['files']
        self.source_priority = file_priority_dict['priority']
        self.combined_df = None
        self.output_file_name = ""


    def read_csv_into_df(self, file_path, source='UNK'):
        df = pd.read_csv(file_path, encoding='utf-8', dtype={'CONS_ID': str, 'CSQ': str, 'CNO': str, 'MNO': str, 'MCD': str, 'AMP': str, 'VLT': str})
        df['SOURCE'] = source
        return df
            
    def concatenate_files(self):
        dataframes = []
        combined_file_name = ""
        for name, file_path in self.file_dict.items():
            df = self.read_csv_into_df(file_path, name)
            dataframes.append(df)
            combined_file_name = combined_file_name + name + "_"
        combined = pd.concat(dataframes, ignore_index=True)
        self.combined_df = self.remove_duplicates_based_on_priority(combined)
        self.output_file_name = combined_file_name
        return self.combined_df
        

    def remove_duplicates_based_on_priority(self, df):
        # Step 2: Identify duplicates based on ['CNO', 'MNO']
        duplicate_mask = df.duplicated(subset=['CNO', 'MNO'], keep=False)
        # Step 3: Store duplicate rows in a new DataFrame
        dups_cno_mno_df = df[duplicate_mask].copy()
        # Step 4: Remove those duplicate rows from the original DataFrame
        df = df[~duplicate_mask].copy()
        # Step 5: Map priority (higher number = higher priority)
        dups_cno_mno_df['_priority'] = dups_cno_mno_df['SOURCE'].map(self.source_priority)

        # Step 6: Sort in descending order of priority and pick the best row
        best_rows_df = (
            dups_cno_mno_df.sort_values(by=['CNO', 'MNO', '_priority'], ascending=[True, True, False])
                        .drop_duplicates(subset=['CNO', 'MNO'], keep='first')
                        .drop(columns=['_priority'])
        )

        # Step 7: Combine and return
        df = pd.concat([df, best_rows_df], ignore_index=True)

        return df