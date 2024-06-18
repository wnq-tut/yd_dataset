import pandas as pd
import random
import os

def count_elements_around_mean(input_list):
    """
    Count the number of elements in the list within and on both sides of the mean +-1 interval.

    :param input_list: input list
    :return: triple (within_range_count, lower_range_count, upper_range_count)
    """
    input_list = [ float(i) for i in input_list]
    mean_value = sum(input_list) / len(input_list)
    within_range_count, lower_range_count, upper_range_count = 0, 0, 0

    for item in input_list:
        if 1.5 <= item <= 4:
            within_range_count += 1
        elif item < 1.5:
            lower_range_count += 1
        elif item > 4:
            upper_range_count += 1
            
    print(f'within_range_count:{within_range_count}\nlower_range_count:{lower_range_count}\nupper_range_count:{upper_range_count}')

    return mean_value

def over_resampling(file, mean, upper_n, lower_n, type):
    new_file = []
    for line in file:
        score1, score2, score3 = line.split('-')[1], line.split('-')[2], line.split('-')[3].split(' ')[0]
        uttid = line.split(' ')[0]
        path = line.split(' ')[1]
        if type == '1':
            score = float(score1)
        elif type == '2':
            score = float(score2)
        elif type == '3':
            score = float(score3)
        if score >= 1.5 and score <= 4:
            new_file.append(f'{line}')
        elif score < 1/5:
            for i in range(lower_n):
                lower_line = f'{i}_{uttid} {path}'
                new_file.append(lower_line)
        elif score > 4:
            for i in range(upper_n):
                upper_line = f'{i}_{uttid} {path}'
                new_file.append(upper_line)
    return new_file


cwd = os.getcwd()
print(cwd)

with open(os.path.join(cwd, 'feat', 'train', 'wav.scp') , "r+", encoding="utf-8") as f:
    file = f.readlines()
    score1_list, score2_list, score3_list = [], [], []
    for line in file:
        score1, score2, score3 = line.split('-')[1], line.split('-')[2], line.split('-')[3].split(' ')[0]
        score1_list.append(score1)
        score2_list.append(score2)
        score3_list.append(score3)
    mean1 = count_elements_around_mean(score1_list)
    mean2 = count_elements_around_mean(score2_list)
    mean3 = count_elements_around_mean(score3_list)
    new_scp1 = over_resampling(file, mean1, 30, 3, '1')
    new_scp2 = over_resampling(file, mean2, 30, 3, '2')
    new_scp3 = over_resampling(file, mean3, 30, 3, '3')
    random.shuffle(new_scp1)
    random.shuffle(new_scp2)
    random.shuffle(new_scp3)
    
with open(os.path.join(cwd, 'over_resampling', 'mis', 'wav.scp'), 'w', encoding="utf-8") as f:
    f.writelines(new_scp1)
with open(os.path.join(cwd, 'over_resampling', 'smooth', 'wav.scp'), 'w', encoding="utf-8") as f:
    f.writelines(new_scp2)
with open(os.path.join(cwd, 'over_resampling', 'total', 'wav.scp'), 'w', encoding="utf-8") as f:
    f.writelines(new_scp3)
    
