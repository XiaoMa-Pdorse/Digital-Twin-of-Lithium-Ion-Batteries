import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys

from importlib import reload
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

data_path = "./"
sys.path.append(data_path)

from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols, CapacityCols
from data_processing.model_data_handler import ModelDataHandler

# Config logging
reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

# Load the cycle and capacity data to memory based on the specified chunk size
# 当前使用 PyBaMM Si 半电池统一仿真数据集（data/si-c-half-cell/，由 si_halfcell_dataset.py
# 生成，DFN(P2D)+热-力耦合，含 SEI+裂纹老化，SOC/SOH 共用，每圈 512 点重采样）
dataset = UniboPowertoolsData(
    test_types=['S'],
    chunk_size=1000000,
    lines=[37, 40],
    charge_line=37,
    discharge_line=40,
    base_path=data_path,
    cyc_path='data/si-c-half-cell/test_result.csv',
    cap_path='data/si-c-half-cell/test_result_trial_end.csv',
    voltage_bounds=(0.0, 2.0)
)

# Prepare the training and testing data for model data handler to load the model input and output data.
# 当前为 Si 半电池统一仿真数据集电池名单：31 只电池（编号 000~030），
# 温度 25.0~40.0 ℃（步长 0.5 ℃，test_name 末段编码 = 温度×100）；每只电池
# 50 圈 C/2 循环（DFN+热-力耦合，含 SEI+裂纹老化，衰减快慢随温度不同）；
# 取 line 37（充电段）曲线作为 CNN 输入，容量表 SOH 列为训练标签。
# 测试电池不含 030：编号最大的电池（其末圈 528 点为全数据集唯一最长圈）需
# 保持在训练名单最后一位（见 si_halfcell_dataset.py 文首补零机制说明）。
_si_cell_names = [f'{i:03d}-SI-3.0-{2500 + 50 * i:04d}-S' for i in range(31)]
_si_test_indices = (5, 11, 17, 23, 29)  # 每 6 只取 1 只做测试（不含末号 030）

train_data_test_names = [name for index, name in enumerate(_si_cell_names)
                         if index not in _si_test_indices]

# UNIBO PowerTools 原始数据集训练电池名单（恢复原数据集时改回 train_data_test_names）
train_data_test_names_unibo = [
    '000-DM-3.0-4019-S', 
    '001-DM-3.0-4019-S', 
    '002-DM-3.0-4019-S', 
    '006-EE-2.85-0820-S', 
    '007-EE-2.85-0820-S', 
    '018-DP-2.00-1320-S', 
    '019-DP-2.00-1320-S',
    '036-DP-2.00-1720-S', 
    '037-DP-2.00-1720-S', 
    '038-DP-2.00-2420-S', 
    '040-DM-4.00-2320-S',
    '042-EE-2.85-0820-S', 
    '045-BE-2.75-2019-S'
]

# Si 半电池仿真数据集测试电池名单
test_data_test_names = [_si_cell_names[index] for index in _si_test_indices]

# UNIBO PowerTools 原始数据集测试电池名单（恢复原数据集时改回 test_data_test_names）
test_data_test_names_unibo = [
    '003-DM-3.0-4019-S',
    '008-EE-2.85-0820-S',
    '039-DP-2.00-2420-S', 
    '041-DM-4.00-2320-S',    
]

dataset.prepare_data(train_data_test_names, test_data_test_names)

# Model data handler will be used to get the model input and output data for further training purpose.
mdh = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE,
])

train_x, train_raw_x, train_y, test_x, test_raw_x, test_y = mdh.get_charge_whole_cycle(soh = True, output_capacity = False, multiple_output=True)

train_y = mdh.keep_only_capacity(train_y, is_multiple_output = True)
test_y = mdh.keep_only_capacity(test_y, is_multiple_output = True)

train_y = train_y[:, [0]]
test_y = test_y[:, [0]]

print(f"Change train_y shape to {train_y.shape}")
print(f"Change test_y shape to {test_y.shape}")

#print(train_x)

# Min-Max Scaler is a popular data normalization
# Xscaled = (X - Xmin) / (Xmax - Xmin)
charge_x_scaler, discharge_x_scaler = mdh.get_scalers()
print(f"charge voltage scaler max_: {charge_x_scaler[0].data_max_}")   
print(f"charge voltage scaler min_: {charge_x_scaler[0].data_min_}")   
print(f"charge current scaler max_: {charge_x_scaler[1].data_max_}")
print(f"charge current scaler min_: {charge_x_scaler[1].data_min_}")   
print(f"charge temperature scaler max_: {charge_x_scaler[2].data_max_}")
print(f"charge temperature scaler min_: {charge_x_scaler[2].data_min_}")   

EXPERIMENT = "cnn_soh_percentage"

experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
print(experiment_name)

# 学习率说明：原 UNIBO 数据集样本量大（步数多），3e-5 可收敛；新 Si 半电池 SOH 数据集
# 仅 1300 个训练样本（26 只训练电池×50 圈），30 epochs 总步数下 3e-5 会使模型输出
# 卡在 0 附近无法收敛（test MAE≈0.93）；同一数据上实测 5e-4 收敛稳定（test MAE≈0.009）。
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

# Model implementation
input = keras.Input(shape=(train_x.shape[1], train_x.shape[2]))

x = layers.Conv1D(64, 32, activation='relu')(input)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Conv1D(64, 32, activation='relu')(x)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Conv1D(64, 32, activation='relu')(x)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Conv1D(64, 32, activation='relu')(x)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
# 输出层 bias 初始化为 1.0（SOH 满值先验）：深层全 ReLU 网络在随机初始化时输出
# 易落在 ReLU 死区（输出恒 0、梯度全零，训练全程停滞——统一数据集首跑曾出现
# val_mae 恒 0.9253 三十个 epoch 不变）；正 bias 使初始输出远离 0，训练稳定收敛。
output = layers.Dense(1, activation='relu',
                      bias_initializer=keras.initializers.Constant(1.0))(x)

model = keras.Model(inputs=input, outputs=output)
model.summary()

# Model compile
model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Setup early stop and check point
es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint(data_path + 'results/trained_model/%s_best.keras' % experiment_name, 
                             save_best_only=True,
                             monitor='val_loss')

history = model.fit(train_x, train_y,
                                epochs=30,
                                batch_size=32,
                                verbose=1,
                                validation_split=0.2,
                                callbacks = [es, mc]
                               )

model.save(data_path + 'results/trained_model/%s.keras' % experiment_name)

hist_df = pd.DataFrame(history.history)
hist_csv_file = data_path + 'results/trained_model/%s_history.csv' % experiment_name
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Load best model
loaded_model = keras.models.load_model(data_path + 'results/trained_model/%s_best.keras' % experiment_name)

# Testing
results = loaded_model.evaluate(test_x, test_y)
print(results)

# Visualiztion
# train loss
fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'],
                    mode='lines', name='train'))
fig.add_trace(go.Scatter(y=history.history['val_loss'],
                    mode='lines', name='validation'))
fig.update_layout(title='Loss trend',
                  xaxis_title='epoch',
                  yaxis_title='loss',
                  width=1400,
                  height=600)
fig.show()

# train dateset prediction result
train_predictions = loaded_model.predict(train_x)
cycle_num = 0
steps_num = train_x.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=train_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH predicted'))
fig.add_trace(go.Scatter(x=step_index, y=train_y.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH actual'))
fig.update_layout(title='Results on training',
                  xaxis_title='Cycle',
                  yaxis_title='SoH percentage',
                  width=1400,
                  height=600)
fig.show()

# test dateset prediction result
test_predictions = loaded_model.predict(test_x)
cycle_num = 0
steps_num = test_x.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=test_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH predicted'))
fig.add_trace(go.Scatter(x=step_index, y=test_y.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH actual'))
fig.update_layout(title='Results on testing',
                  xaxis_title='Cycle',
                  yaxis_title='SoH percentage',
                  width=1400,
                  height=600)
fig.show()
