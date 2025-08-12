def check_t_1_c_1(row):
    return int(row['ELC'] == 'N' and 'SP' in row['ent_cd_list'][-3:])

def check_t_1_c_2(row):
    return int(row['ELC'] == 'N' and len(row['adv_ut_list']) > 0 and row['adv_ut_list'][-1] < 10)

def check_t_1_c_3(row):
    return int(row['ELC'] == 'N' and 'LW' in row['ent_cd_list'][-3:])

def check_t_2_c_1(row):
    if row['ELC'] == 'Y' and len(row['adv_ut_list']) >= 7:
        last = row['adv_ut_list'][-1]
        prev_6 = row['adv_ut_list'][-7:-1]
        avg = sum(prev_6) / 6
        return int(avg > 50 and last > 4 * avg)
    return 0

def check_t_2_c_2(row):
    if row['ELC'] == 'Y' and len(row['adv_ut_list']) >= 7:
        last = row['adv_ut_list'][-1]
        prev_6 = row['adv_ut_list'][-7:-1]
        avg = sum(prev_6) / 6
        return int(avg > 50 and last > 8 * avg)
    return 0

def check_t_2_c_3(row):
    return int(row['ELC'] == 'Y' and 'NR' in row['ent_cd_list'][-3:])

# def check_t_2_c_4(row):
#     if row['ELC'] == 'Y':
#         try:
#             mno = int(row['MNO'])
#             return int(7000001 <= mno <= 7097000)
#         except (ValueError, TypeError):
#             return 0
#     return 0

def check_t_a_c_1(row):
    return int(len(row['ent_cd_list']) > 0 and row['ent_cd_list'][-1] in {'LCC', 'NR', 'Read'})