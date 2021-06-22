import pandas as pd
import numpy as np
import os


def merge_lstm_and_env(lstm_file, env_file, No):
    lstm_path = 'LSTM_Trace/' + lstm_file
    env_path = 'Env_Trace/' + env_file
    lstm_trace = pd.read_csv(lstm_path)
    env_trace = pd.read_excel(env_path)
    merged_trace = pd.concat([lstm_trace, env_trace.iloc[:, 2:]], axis=1)
    merged_trace.columns = ['本车_y', '本车_x', '0',
                            '目标车道前车_y', '目标车道前车_x', '0',
                            '目标车道后车_y', '目标车道后车_x', '0',
                            '原车道前车_y', '原车道前车_x', '0',
                            '原车道后车_y', '原车道后车_x']
    merged_trace.drop(columns=['0'], axis=1, inplace=True)
    merged_trace.to_csv('merged_trace/' + '{}_merged_trace.csv'.format(No), index=0)


def process_data(data_name, No):
    baseXPos = 375.5
    baseZPos = -165.0
    roadWidth = 8.0
    time_step = 0.02
    import_path = ''
    import_path = import_path + data_name
    # raw_data = pd.read_excel(import_path)
    raw_data = pd.read_csv(import_path)
    raw_data.dropna(axis=1, inplace=True)
    meanGoalX = (raw_data.loc[:, '目标车道前车_y'].sum() + raw_data.loc[:, '目标车道后车_y'].sum()) / (2 * len(raw_data))
    meanOrigX = (raw_data.loc[:, '原车道前车_y'].sum() + raw_data.loc[:, '原车道后车_y'].sum()) / (2 * len(raw_data))
    for col in raw_data.columns:
        if '_y' in col:
            raw_data.loc[:, col] = raw_data.loc[:, col].apply(
                lambda x: baseXPos - roadWidth * (x - meanOrigX) / (meanGoalX - meanOrigX))
        else:
            raw_data.loc[:, col] = raw_data.loc[:, col].apply(lambda x: baseZPos + x)
    data = pd.DataFrame(columns=['d_pre_goal', 'd_tail_goal', 'd_pre_raw', 'interval_pre', 'interval_tail',
                                 'v_x', 'v_y', 'delta_v_goal', 'delta_v_raw', 'delta_a_x', 'delta_a_y',
                                 'a_x', 'a_y'])
    data['d_pre_goal'] = (raw_data['目标车道前车_x'] - raw_data['本车_x']) * 1.5
    data['d_tail_goal'] = (raw_data['本车_x'] - raw_data['目标车道后车_x']) * 1.5
    data['d_pre_raw'] = (raw_data['原车道前车_x'] - raw_data['本车_x']) * 1.5
    data['interval_pre'] = (raw_data['本车_y'] - raw_data['目标车道前车_y']) * 1.5
    data['interval_tail'] = (raw_data['本车_y'] - raw_data['目标车道后车_y']) * 1.5
    lastRow = None
    for i, row in raw_data.iterrows():
        if i == 0:
            lastRow = row
            continue
        v_x = (row['本车_x'] - lastRow['本车_x']) * 1.5 / time_step
        v_pre_goal_x = (row['目标车道前车_x'] - lastRow['目标车道前车_x']) * 1.5 / time_step
        v_pre_raw_x = (row['原车道前车_x'] - lastRow['原车道前车_x']) * 1.5 / time_step
        v_y = (lastRow['本车_y'] - row['本车_y']) * 1.5 / time_step
        delta_v_goal = v_x - v_pre_goal_x
        delta_v_raw = v_x - v_pre_raw_x
        lastRow = row
        data.loc[i, 'v_x'] = v_x
        data.loc[i, 'v_y'] = v_y
        data.loc[i, 'delta_v_goal'] = delta_v_goal
        data.loc[i, 'delta_v_raw'] = delta_v_raw
    data.loc[0, 'v_x': 'delta_v_raw'] = data.loc[1, 'v_x': 'delta_v_raw']

    for i, row in data.iterrows():
        if i == 0:
            lastRow = row
            continue
        a_x = (row['v_x'] - lastRow['v_x']) / time_step
        a_y = (row['v_y'] - lastRow['v_y']) / time_step
        data.loc[i, 'a_x'] = a_x
        data.loc[i, 'a_y'] = a_y
    data.loc[0, 'a_x': 'a_y'] = data.loc[1, 'a_x': 'a_y']

    for i, row in data.iterrows():
        if i == 0:
            lastRow = row
            continue
        delta_a_x = (row['a_x'] - lastRow['a_x']) / time_step
        delta_a_y = (row['a_y'] - lastRow['a_y']) / time_step
        data.loc[i, 'delta_a_x'] = delta_a_x
        data.loc[i, 'delta_a_y'] = delta_a_y
    data.loc[0, 'delta_a_x': 'delta_a_y'] = data.loc[1, 'delta_a_x': 'delta_a_y']
    data.to_csv('train_data/' + '{}_train_data.csv'.format(No), index=0)


if __name__ == '__main__':
    lstm_files = []
    env_files = []
    for file in os.listdir('LSTM_Trace'):
        lstm_files.append(file)
    lstm_files.sort()
    for file in os.listdir('Env_Trace'):
        env_files.append(file)
    env_files.sort()

    assert len(lstm_files) == len(env_files), 'lstm_trace not matching env_trace'
    for i in range(len(lstm_files)):
        no = lstm_files[i].split('_')[0]
        merge_lstm_and_env(lstm_files[i], env_files[i], no)

    for file in os.listdir('merged_trace'):
        no = file.split('_')[0]
        process_data('merged_trace/' + file, no)

