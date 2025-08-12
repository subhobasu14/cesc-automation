import os

PATH = os.getenv('FILE_PATH_1', '/opt/airflow/data/input/')

meter_defect_range_file_name = "{}DEFECTIVE_METERS_2PIRAD_2ND PHASE_JAN 2023_2PIRAD.csv".format(PATH)
mcd_code_file_name = '{}m_cod.csv'.format(PATH)

# meter_defect_range_file_name = '../data/input/DEFECTIVE METERS_2PIRAD_2ND PHASE_JAN 2023_2PIRAD.csv'
# mcd_code_file_name = '../data/input/m_cod.csv'

consumer_columns = ['CONS_ID', 'CSQ', 'CNO', 'MNO', 'MCD', 'FDT', 'XCHG', 'ADV']
consumption_columns = ['CONS_ID', 'CSQ', 'MNO', 'BC_CD', 'BL_CD', 'ENT_CD', 'ADV_UT']

consumers_file_column_dtype = {'CONS_ID': str, 'CNO': str, 'CSQ': str, 'MNO': str}
consumptions_file_column_dtype = {'CONS_ID': str, 'CSQ': str, 'MNO': str}
csv_file_delimeter = '|'
consumer_file_supported_rows = 10000
consumptions_file_supported_rows = consumer_file_supported_rows * 36
output_filename_prefix = 'proactive_prediction'

supported_mcd_values = [102, 103, 112, 114, 120, 121, 125, 126, 140, 141, 142, 143, 145, 146, 147, 150, 151, 152, 153, 155, 212, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 421, 423, 425, 431, 432, 515, 516, 549, 550, 551, 560, 561, 562, 564, 565, 566, 567, 568, 569, 571, 572, 573, 574, 575, 576, 577, 578, 580, 581, 582, 585, 586, 587, 589, 590, 591, 595, 596, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 617, 621, 634, 637, 639, 641, 680, 681, 682, 800, 802, 803, 804, 805, 806, 807, 809, 812, 815, 821, 823, 824, 825, 827, 828, 829, 834, 836, 837, 838, 839, 840, 841, 855, 857, 869, 870, 884, 896, 898, 899, 901, 905, 906, 907, 911, 912, 917, 918]

rejection_reason_column_name = 'REMARKS'

consumption_nan_to_val_cutoff_ratio = 0.6
consumption_min_count = 5

model_path = './model/pp_catboost_model_02012025_1.cbm'

error_code_for_exception = 400

prediction_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
prediction_category = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7',
          '0.7-0.8', '0.8-0.9', '0.9-1.0']

row_filter_errors = {
	  "missing_fdt" : "Consumer without FDT value.",
	  "different_mno" : "Consumer with different MNO in Master & Consumption.",
	  "no_consumption" : "Consumer with no consumption data.",
	  "defect_range" : "Consumer with MNO in Defect Range.",
	  "less_consumption" : "Consumer with less than 4 consumption.",
	  "nonsupported_mcd_values" : "Consumer with non supported MCD value."
  }

from features.MeterDefectRange import MeterDefectRange

# Instantiate the shared variable once at startup.
meter_defect_range = MeterDefectRange(meter_defect_range_file_name, mcd_code_file_name)

RULES_WITH_WEIGHTS = {
    'check_t_1_c_1': 4, # Electromechanical Meter and SP in Excp Code in any of the last 3 months.
    'check_t_1_c_2': 6, # Electromechanical Meter and Last month consumption <10 units.
    'check_t_1_c_3': 5, # Electromechanical Meter and LW exists in Excp Code in any of the last 3 months.
    'check_t_2_c_1': 8, # Electronic Meter and Last month consumption> 4 times the avg consumption of last 6 months (excluding the last month consumption)
    					# &
    					# Avg consumption of last 6 months (excluding the last month consumption) > 50 units
    'check_t_2_c_2': 10,# Electronic Meter and Last month consumption> 8 times the avg consumption of last 6 months (excluding the last month consumption)
    					# &
    					# Avg consumption of last 6 months (excluding the last month consumption) > 50 units
    'check_t_2_c_3': 4, # Electronic Meter and NR in Excp Code in any of the last 3 months
    'check_t_2_c_4': 3, # Electronic Meter and If the meter serial number lies in the range:  7000001 - 7097000
    'check_t_a_c_1': 7, # Electromechanical & Electronic Meter and if the meter inspector reports any of the following cases during last reading: LCC, NR, Read
}

