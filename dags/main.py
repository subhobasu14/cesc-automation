import numpy as np
import json
import pandas as pd
import ast
import os
from airflow.decorators import dag, task, task_group
from datetime import datetime
import mlflow
import mlflow.sklearn
import catboost as cb
# import sys
# sys.path.append('../')
# import sys
# sys.path.append('/opt/airflow/features')
from features.CombineInputFiles import CombineConsumerAndConsumption
from features.CombineInputFiles import CombineConsumerAndMeterHistory
from features.ConcatenateFiles import FileConcatenator
from features.PrepareModelInput import PrepareModelInputData
from features.CatboostModelTrainer import TrainCatboostModel

PATH = os.getenv('FILE_PATH_2', '/opt/airflow/input/')
config_env = os.getenv('CONFIG_ENV', 'docker')
config_file_name = f"config.{config_env}.json"

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("CatBoostClassifier")
# mlflow.catboost.autolog()

print("PATH ========================= ", '{}{}'.format(PATH, config_file_name))
def replace_nan_with_none(lst):
    return [None if isinstance(x, float) and np.isnan(x) else x for x in lst]


def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

with open('{}{}'.format(PATH, config_file_name), 'r') as f:
        steps = json.load(f)
        
@task
def processRawData():
    print('Processing data')
    combineFiles = steps['combine']
    output_directory = combineFiles['output_directory']
    combineFiles_datasets = combineFiles['files']
    for dataset_name, details in combineFiles_datasets.items():
        master_file = details['consumer']
        consumption_file = details['consumption']
        report_file = None
        custom_process = None
        if 'report' in details:
            report_file = details['report']

        if ('custom_processing' in details) and (details['custom_processing']):
            custom_process = True

        if custom_process:
            dp_obj = CombineConsumerAndMeterHistory(master_file, consumption_file)
            df = dp_obj.get_combined_raw_data()
        else:
            dp_obj = CombineConsumerAndConsumption(master_file, consumption_file, report_file)
            df = dp_obj.get_combined_raw_data()

        df['adv_ut_list'] = df['adv_ut_list'].apply(replace_nan_with_none) # type: ignore
        # df['ent_cd_list'] = df['ent_cd_list'].apply(safe_parse_list_of_strings)

        output_file = output_directory + dataset_name + "_raw.csv"
        print('output path', output_file)
        df[['CONS_ID', 'CSQ', 'CNO', 'MNO', 'MCD', 'MFG', 'AMP', 'VLT', 'SUP', 'SMT', 'ELC', 'adv_ut_list', 'ent_cd_list', 'defect_range', 'LABEL']].to_csv(output_file, index=False) # type: ignore
        print('Dataset {} completed'.format(dataset_name))
        
    print('Data processing completed')


### Concatenate Files
@task
def concatenateFiles():
    print('File concatenation started')
    concatenate_file_dict = steps['concatenate']
    output_directory = concatenate_file_dict['output_directory']
    concatenateFileObject = FileConcatenator(concatenate_file_dict)
    combined_df = concatenateFileObject.concatenate_files()

    combined_output_file = output_directory + concatenateFileObject.output_file_name + ".csv"
    combined_df.to_csv(combined_output_file, index=False)
    print('Concatenation completed')


### Prepare Model Input
@task
def prepareModelInput():
    print('Model Input preparation started')
    model_input_dict = steps['prepare_model_input']
    output_directory = model_input_dict['output_directory']
    combined_file_path = model_input_dict['files']
    columns_to_use = ['CONS_ID', 'CSQ', 'CNO', 'MNO', 'MCD', 'MFG', 'AMP', 'VLT', 'SUP',
        'SMT', 'ELC', 'adv_ut_list', 'ent_cd_list', 'defect_range', 'LABEL']

    columns_dtype = {'CONS_ID': str, 'CSQ': str, 'CNO': str, 'MNO': str, 'MCD': str, 'AMP': str, 'VLT': str}


    mi_df = pd.read_csv(combined_file_path, encoding = 'utf-8', dtype = columns_dtype, usecols = columns_to_use) # type: ignore
    mi_df['adv_ut_list'] = mi_df['adv_ut_list'].apply(ast.literal_eval)
    mi_df['ent_cd_list'] = mi_df['ent_cd_list'].apply(ast.literal_eval)

    model_input_obj = PrepareModelInputData(mi_df)
    model_input_df = model_input_obj.prepare_data()

    model_fea_output_file_path = output_directory + get_filename_without_extension(combined_file_path) + "model_input_data.csv"
    model_input_df.to_csv(model_fea_output_file_path, index=False)
    print('Model Input preparation DONE!')
    
@task
def prepareModelInput_2():
    print('Model Input 2 preparation started')
    from sklearn.preprocessing import LabelEncoder

    lb = LabelEncoder() 
    
    model_input_dict = steps['prepare_model_input']
    output_directory = model_input_dict['output_directory']
    combined_file_path = model_input_dict['files']
    columns_to_use = ['CONS_ID', 'CSQ', 'CNO', 'MNO', 'MCD', 'MFG', 'AMP', 'VLT', 'SUP',
        'SMT', 'ELC', 'adv_ut_list', 'ent_cd_list', 'defect_range', 'LABEL']

    columns_dtype = {'CONS_ID': str, 'CSQ': str, 'CNO': str, 'MNO': str, 'MCD': str, 'AMP': str, 'VLT': str}

    mi_df = pd.read_csv(combined_file_path, encoding = 'utf-8', dtype = columns_dtype, usecols = columns_to_use) # type: ignore
    mi_df['adv_ut_list'] = mi_df['adv_ut_list'].apply(ast.literal_eval)
    mi_df['ent_cd_list'] = mi_df['ent_cd_list'].apply(ast.literal_eval)

    model_input_obj = PrepareModelInputData(mi_df)
    model_input_df = model_input_obj.prepare_data()

    model_input_df['ELC'] = lb.fit_transform(model_input_df['ELC'])
    model_fea_output_file_path = output_directory + get_filename_without_extension(combined_file_path) + "model_input_data_2.csv"
    model_input_df.to_csv(model_fea_output_file_path, index=False)
    print('Model Input 2 preparation DONE!')
    
@task
def trainModel1():
    model_train_dict = steps['train_catboost_model']
    output_directory = model_train_dict['output_directory']
    model_training_file = model_train_dict['training_data']
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("CatBoostClassifier")
    train_df = pd.read_csv(model_training_file, usecols = TrainCatboostModel.FEATURE_COLUMNS, dtype = TrainCatboostModel.COLUMNS_DTYPE) # type: ignore
    
    # Create a Dataset object
    dataset = mlflow.data.from_pandas(
        train_df, source=model_training_file, name= "CESC Training Data"
    )
    
    cbmt = TrainCatboostModel(train_df, output_directory)
    with mlflow.start_run(run_name="CatBoost Classifier with all features") as run:
        model = cbmt.train_and_save_model()
        mlflow.sklearn.log_model(model, "CatBoostClassifierModel")
        mlflow.register_model(
            "runs:/{}/CatBoostClassifierModel".format(run.info.run_id), 
            "CatBoostClassifierModel"
        )
        mlflow.log_input(dataset)
        params = model.get_params()
        print("Params: ", params)
        eval_results = model.get_evals_result()
        print(f"eval_results=======: {eval_results}")
        f1_learn = eval_results['learn']['F1'][-1]
        f1_validation = eval_results['validation']['F1'][-1]
        logloss_learn = eval_results['learn']['Logloss'][-1]
        logloss_validation = eval_results['validation']['Logloss'][-1]
        
        mlflow.log_param("Learning Rate", params['learning_rate'])
        mlflow.log_param("Depth", params['depth'])
        mlflow.log_param("Border Count", params['border_count'])
        mlflow.log_param("Class Weights", params['class_weights'])
        
        mlflow.log_metric("f1_learn", f1_learn)
        mlflow.log_metric("f1_validation", f1_validation)
        mlflow.log_metric("logloss_learn", logloss_learn)
        mlflow.log_metric("logloss_validation", logloss_validation)
        
    cbmt.feature_selection_pipeline()
    
@task
def trainModel2():
    print('Training model 2')

@task
def trainModel3():
    print('Training model 3')

@task
def trainModel4():
    print('Training model 4')
    
@dag(start_date=datetime(2023,1,1), schedule=None, catchup=False)
def feature_engineering():
    print("Inside DAG ================================")
    # processRawData() >> concatenateFiles()  # type: ignore
    # concatenateFiles() >> prepareModelInput() # type: ignore
    # concatenateFiles() >> prepareModelInput_2() # type: ignore
    # @task_group(group_id='dependent_tasks')
    # def run_dependent_tasks():
    #     processRawData() >> concatenateFiles() # type: ignore
        
    # @task_group(group_id='independent_tasks')
    # def run_independent_tasks():
    #     prepareModelInput() >> [trainModel1(), trainModel2()] # type: ignore
    #     prepareModelInput_2() >> [trainModel3(), trainModel4()] # type: ignore
        
    # processRawData() >> concatenateFiles() >> [prepareModelInput(), prepareModelInput_2()] # type: ignore
    
    # processRawData() >> concatenateFiles() >> [prepareModelInput() >> [trainModel1(), trainModel2()], prepareModelInput_2() >> [trainModel3(), trainModel4()]] # type: ignore
    
    processRawData() >> concatenateFiles() >> prepareModelInput() >> trainModel1() # type: ignore
    # trainModel1()
    # run_dependent_tasks() >> run_independent_tasks() # type: ignore
    
feature_engineering()

