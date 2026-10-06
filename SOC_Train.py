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
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols, CapacityCols
from data_processing.model_data_handler import ModelDataHandler

TIME_STEPS = 8
# 统一数据集每条半循环曲线 512 点（旧 50 点 SOC 数据的 10 倍密度），滑窗按
# SEQ_STRIDE 步进采样，使样本规模与训练时长与旧数据集相当；设为 1 可恢复
# 逐点稠密采样（样本数与训练时间将增大约 8 倍）。
SEQ_STRIDE = 8

def create_sequence_data(data_x, data_y):
    seq_data_x = []
    seq_data_y = []

    for i in range(0, len(data_x) - TIME_STEPS, SEQ_STRIDE):
        # Ensure the current is not zero at the first and last time steps of the sequence to prevent the sequence from spanning discharge cycles.        
        if data_x[i][0] !=0 and data_x[i+TIME_STEPS-1][0] != 0:
            seq_data_x.append(data_x[i:i+TIME_STEPS])
            seq_data_y.append(data_y[i+TIME_STEPS])

    seq_data_x = np.array(seq_data_x)
    seq_data_y = np.array(seq_data_y)

    return seq_data_x, seq_data_y

data_path = "./"
sys.path.append(data_path)

# Config logging
reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

# Load the cycle and capacity data to memory based on the specified chunk size
# 当前使用 PyBaMM Si 半电池统一仿真数据集（data/si-c-half-cell/，由 si_halfcell_dataset.py
# 生成，DFN(P2D)+热-力耦合，含 SEI+裂纹老化，SOC/SOH 共用）；恢复 UNIBO PowerTools
# 原始数据集时删除下方 cyc_path/cap_path/voltage_bounds 三个参数即可。
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
# 50 圈 C/2 循环（DFN+热-力耦合，含 SEI+裂纹老化，容量表 SOH 列携带真实衰减信息）。
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
#    '002-DM-3.0-4019-S', 
    '006-EE-2.85-0820-S', 
    '007-EE-2.85-0820-S', 
    '018-DP-2.00-1320-S', 
    '019-DP-2.00-1320-S',
#    '036-DP-2.00-1720-S', 
#    '037-DP-2.00-1720-S', 
#    '038-DP-2.00-2420-S', 
#    '040-DM-4.00-2320-S',
    '042-EE-2.85-0820-S', 
    '045-BE-2.75-2019-S'
]

# Si 半电池仿真数据集测试电池名单（升序；编号最大的电池 030 位于训练名单末尾，
# 测试名单内每圈均有 >= 16 行补零断隔，满足时序滑窗不跨循环）
test_data_test_names = [_si_cell_names[index] for index in _si_test_indices]

# UNIBO PowerTools 原始数据集测试电池名单（恢复原数据集时改回 test_data_test_names）
test_data_test_names_unibo = [
    '003-DM-3.0-4019-S',
#    '008-EE-2.85-0820-S',
#    '039-DP-2.00-2420-S', 
#    '041-DM-4.00-2320-S',    
]

dataset.prepare_data(train_data_test_names, test_data_test_names)

# Model data handler will be used to get the model input and output data for further training purpose.
mdh = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE,
], [CapacityCols.SOH])

train_x, train_raw_x, train_y, test_x, test_raw_x, test_y = mdh.get_discharge_whole_cycle(soh = False, output_capacity = False)

train_y = mdh.keep_only_capacity(train_y, is_multiple_output = True)
test_y = mdh.keep_only_capacity(test_y, is_multiple_output = True)

# Flatten train/test dataset to create time sequence dataset
train_x_flat = train_x.reshape(-1, train_x.shape[2])
train_x_raw_flat = train_raw_x.reshape(-1, train_raw_x.shape[2])
train_y_flat = train_y.reshape(-1, 1)
test_x_flat = test_x.reshape(-1, test_x.shape[2])
test_x_raw_flat = test_raw_x.reshape(-1, test_raw_x.shape[2])
test_y_flat = test_y.reshape(-1, 1)

train_x_seq, train_y_seq = create_sequence_data(train_x_flat, train_y_flat)
train_raw_x_seq, _ = create_sequence_data(train_x_raw_flat, train_y_flat)
test_x_seq, test_y_seq = create_sequence_data(test_x_flat, test_y_flat)
test_raw_x_seq, _ = create_sequence_data(test_x_raw_flat, test_y_flat)

charge_x_scaler, discharge_x_scaler = mdh.get_scalers()
print(f"discharge voltage scaler max_: {discharge_x_scaler[0].data_max_}")   
print(f"discharge voltage scaler min_: {discharge_x_scaler[0].data_min_}")   
print(f"discharge current scaler max_: {discharge_x_scaler[1].data_max_}")
print(f"discharge current scaler min_: {discharge_x_scaler[1].data_min_}")   
print(f"discharge temperature scaler max_: {discharge_x_scaler[2].data_max_}")
print(f"discharge temperature scaler min_: {discharge_x_scaler[2].data_min_}")   
print(f"discharge SOH scaler max_: {discharge_x_scaler[3].data_max_}")
print(f"discharge SOH scaler min_: {discharge_x_scaler[3].data_min_}")   

EXPERIMENT = "lstm_soc_percentage"

experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
print(experiment_name)

opt = tf.keras.optimizers.Adam(learning_rate=0.00003)

# Model implementation
model = Sequential()
model.add(LSTM(256,
                return_sequences=True,
                unroll=True,
                input_shape=(TIME_STEPS, train_x.shape[2])))
model.add(Dropout(0.2)) # Dropout 20% of neurons to prevent overfitting.

model.add(LSTM(256, unroll=True, return_sequences=True))
model.add(Dropout(0.2)) # Dropout 20% of neurons to prevent overfitting.

model.add(LSTM(128, unroll=True, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint(data_path + 'results/trained_model/%s_best.keras' % experiment_name, 
                             save_best_only=True, 
                             monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001) # Halve the learning rate after 5 epochs of stagnation.

history = model.fit(train_x_seq, train_y_seq,
                                epochs=30,
                                batch_size=128,
                                verbose=1,
                                validation_split=0.2,
                                callbacks = [es, mc, reduce_lr]
                               )

model.save(data_path + 'results/trained_model/%s.keras' % experiment_name)

hist_df = pd.DataFrame(history.history)
hist_csv_file = data_path + 'results/trained_model/%s_history.csv' % experiment_name
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Load best model
loaded_model = keras.models.load_model(data_path + 'results/trained_model/%s_best.keras' % experiment_name)

# Testing
results = loaded_model.evaluate(test_x_seq, test_y_seq)
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
train_predictions = loaded_model.predict(train_x_seq)
cycle_num = 0
steps_num = train_x_seq.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=train_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC predicted'))
fig.add_trace(go.Scatter(x=step_index, y=train_y_seq.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC actual'))
fig.update_layout(title='Results on training',
                  xaxis_title='Sample',
                  yaxis_title='SoC percentage',
                  width=1400,
                  height=600)
fig.show()

# test dateset prediction result
test_predictions = loaded_model.predict(test_x_seq)
cycle_num = 0
steps_num = test_x_seq.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=test_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC predicted'))
fig.add_trace(go.Scatter(x=step_index, y=test_y_seq.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC actual'))
fig.update_layout(title='Results on testing',
                  xaxis_title='Sample',
                  yaxis_title='SoC percentage',
                  width=1400,
                  height=600)
fig.show()

# Convert to INT8 tflite model.
def representative_dataset():
    # 量化校准数据须为 float32（MinMaxScaler 数据管线输出 float64，需显式转换）
    for input_value in tf.data.Dataset.from_tensor_slices(train_x_seq.astype(np.float32)).batch(1).take(1000):
        yield [input_value]

# 本机为 TF 2.16 + 独立 Keras 3（keras 3.15）：from_keras_model 转换该 Keras 3 LSTM 模型
# 会在 MLIR 阶段发生 native 崩溃（LLVM ERROR: Failed to infer result type(s)，
# ReadVariableOp 缺失 'value' 属性）；改用"冻结变量为常量的具体函数"方式转换。
@tf.function(input_signature=[tf.TensorSpec([None, TIME_STEPS, train_x_seq.shape[2]], tf.float32)])
def soc_serving(x):
    return loaded_model(x)

soc_concrete = soc_serving.get_concrete_function()
soc_frozen = convert_variables_to_constants_v2(soc_concrete)

converter = tf.lite.TFLiteConverter.from_concrete_functions([soc_frozen])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

# Save the model.
with open('results/trained_model/BMS_SOC_INT8.tflite', 'wb') as f:
  f.write(tflite_quant_model)

def export_numpy_to_c_header(x_array, y_array, filename="test_data.h"):
    """
    Convert the NumPy array and write it to the C Header file.
    """
    print(f"Data is being exported to {filename} ...")
    
    with open(filename, 'w') as f:
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")

        # Write the Normalize information of the input data for easy reference during C language development.
        f.write(f"// Normalize scale factor(voltage, current, temperature, SOH)\n")
        scale_max_str = f"{discharge_x_scaler[0].data_max_[0]}, {discharge_x_scaler[1].data_max_[0]}, {discharge_x_scaler[2].data_max_[0]}, {discharge_x_scaler[3].data_max_[0]}"
        scale_min_str = f"{discharge_x_scaler[0].data_min_[0]}, {discharge_x_scaler[1].data_min_[0]}, {discharge_x_scaler[2].data_min_[0]}, {discharge_x_scaler[3].data_min_[0]}"
        f.write(f"const float normalize_scale_max[] = {{{scale_max_str}}};\n")
        f.write(f"const float normalize_scale_min[] = {{{scale_min_str}}};\n")

        def write_array_to_c(arr, array_name):
            slice_start = 0  # Adjust the starting position of the slice according to actual needs.
            slice_size = 64  # Adjust the slice size according to actual needs.
            slice_arr = arr[slice_start: slice_start + slice_size, ...]

            flat_arr = slice_arr.flatten()
            length = len(flat_arr)

            # Write the array dimensions for easy reference during C language development
            f.write(f"// Original array shape: {slice_arr.shape}\n")
            f.write(f"const int {array_name}_dim[] = {{{', '.join(map(str, slice_arr.shape))}}};\n")
            f.write(f"const int {array_name}_length = {length};\n\n")
            
            # Declare the C array (using float as an example)
            f.write(f"const float {array_name}[{length}] = {{\n")
            
            # Write the values ​​in batches to avoid compiler errors caused by single lines being too long (32 values ​​per line).
            for i in range(0, length, 32):
                chunk = flat_arr[i:i+32]
                chunk_str = ", ".join([f"{val:.6f}" for val in chunk])
                if i + 32 < length:
                    f.write(f"    {chunk_str},\n")
                else:
                    f.write(f"    {chunk_str}\n")
            f.write("};\n\n")

        # Write the X and Y data
        write_array_to_c(x_array, "test_x_seq")
        write_array_to_c(y_array, "test_y_seq")

        f.write("#endif // TEST_DATA_H\n")
    
    print("Export completed!")

# Export test_raw_x_seq and test_y_seq data to C header file for later use in C language development.
export_filepath = data_path + 'results/trained_model/SOC_test_data.h'
export_numpy_to_c_header(test_raw_x_seq, test_y_seq, filename=export_filepath)
