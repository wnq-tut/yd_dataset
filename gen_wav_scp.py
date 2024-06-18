import pandas as pd
import random

def read_excel(file_path):
    # Read Excel Files
    xls = pd.ExcelFile(file_path)

    # Get the names of all sheet pages
    sheet_names = xls.sheet_names

    # Save the contents of each sheet page into three matrices
    matrix_sheet1 = None
    matrix_sheet2 = None
    matrix_sheet3 = None

    for idx, sheet_name in enumerate(sheet_names):
        # Read the data of each sheet page
        df = pd.read_excel(file_path, sheet_name, index_col=None)

        # Convert DataFrame to Matrix (2D Array)
        matrix = df.values.tolist()

        # Save to three matrices respectively
        if idx == 0:
            matrix_sheet1 = matrix
        elif idx == 1:
            matrix_sheet2 = matrix
        elif idx == 2:
            matrix_sheet3 = matrix

    return matrix_sheet1, matrix_sheet2, matrix_sheet3

def get_score(x, y, z):
    i = 1
    if x == 0:
        i = i - 1
    if y == 0:
        i = i - 1
    if z == 0:
        i = i - 1
    score = round(((x+y+z)/i), 2)
    # score = x+y+z
    return score

# Specify the Excel file path
file_path = 'data_pre\score.xlsx'

# Call function to read Excel file content
matrix_sheet1, matrix_sheet2, matrix_sheet3 = read_excel(file_path)
all = []
trains = []
vals = []
for i in range(0, len(matrix_sheet1)):
    line = ''
    name = matrix_sheet1[i][0]
    score1 = get_score(matrix_sheet1[i][1], matrix_sheet1[i][2], matrix_sheet1[i][3])
    score2 = get_score(matrix_sheet2[i][1], matrix_sheet2[i][2], matrix_sheet2[i][3])
    score3 = get_score(matrix_sheet3[i][1], matrix_sheet3[i][2], matrix_sheet3[i][3])
    utt_id = f'{name}-{score1}-{score2}-{score3}'
    line = f'{utt_id} /home/you/workspace/yd_dataset/yd/{name[:6]}/{name}.wav'
    all.append(line)
random.shuffle(all)
index = int(len(all)*0.9)
trains = all[:index]
vals = all[index:]

with open("wav_train.scp", "w", encoding="utf-8") as t, \
     open("wav_val.scp", "w", encoding="utf-8") as v:
    for line in trains:
        t.write(f'{line}\n')
    for line in vals:
        v.write(f'{line}\n')
        
