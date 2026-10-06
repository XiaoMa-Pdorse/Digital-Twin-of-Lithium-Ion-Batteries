# -*- coding: utf-8 -*-
"""Step 3：LSTM 预测容量衰减曲线与剩余寿命（RUL）—— 硅基负极数字孪生原型（第 3/4 步）。

功能
----
1. 读取统一仿真数据集（data/si-c-half-cell/，DFN+热-力耦合，含 SEI 与
   裂纹老化）中 31 只电池的逐圈充电容量，归一化为 SOH 曲线（首圈 = 1.0），
   构造“前 20 圈 → 后 30 圈”的序列到序列样本（与展示流程“输入前 20 次循环
   容量数据，预测后续衰减曲线”一致）；
2. 数据增强：对每只电池的曲线做时间轴缩放（模拟不同温度/倍率下衰减快慢不同
   的电池），样本量 31 -> 155；
3. 训练小型 LSTM：输入 20 点 SOH 序列 -> 输出未来 30 圈衰减曲线；
4. RUL 估算：对预测曲线末端做线性外推，计算到达 EOL 阈值（默认 SOH = 80%）
   的循环数，与真实曲线外推值对比；
5. 输出 results/digital_twin/：模型 rul_lstm.keras、训练历史 CSV、
   指标 JSON、演示图 rul_demo.png（供 Step 4 的 Streamlit 直接加载）。

面向 Si 基材料的泛化设计
------------------------
- 模型输入仅含容量（SOH）序列本身，不含温度/型号等专有标签：任何 Si 体系
  电池只要提供前 20 圈容量即可套用（同构输入，可平滑迁移 NASA 类公开数据）；
- 数据增强覆盖"衰减快慢"分布，测试集为未见过的高温电池组（衰减最快），
  检验模型对更陡衰减曲线的外推能力；
- 归一化按首圈容量（SOH 语义），与电池绝对容量解耦。

用法
----
python rul_lstm.py        # 完整训练 + 评估 + 出图（CPU 约 1~2 分钟）
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免弹窗阻塞（app.py 同时复用本模块）
import matplotlib.pyplot as plt
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SOH_TABLE_PATH = os.path.join(REPO_ROOT, 'data', 'si-c-half-cell',
                              'test_result_trial_end.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'digital_twin')
MODEL_PATH = os.path.join(OUT_DIR, 'rul_lstm.keras')
HISTORY_CSV_PATH = os.path.join(OUT_DIR, 'rul_lstm_history.csv')
METRICS_JSON_PATH = os.path.join(OUT_DIR, 'rul_metrics.json')
RUL_PNG_PATH = os.path.join(OUT_DIR, 'rul_demo.png')

IN_LEN = 20                # 输入窗口：前 20 圈
OUT_LEN = 30               # 预测窗口：其后 30 圈
SCALES = (0.85, 0.925, 1.0, 1.075, 1.15)   # 时间轴缩放增强系数
EOL_THRESHOLD = 0.80       # EOL 判定：SOH 降至 80%
TAIL_FIT_POINTS = 15       # RUL 线性外推的末端拟合点数
EPOCHS = 300
BATCH_SIZE = 32
SEED = 42

# 划分：后 10 只（高温、衰减最快）留作泛化测试
TEST_CELL_COUNT = 10
VAL_CELL_COUNT = 5

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 数据加载与样本构造
# ---------------------------------------------------------------------------
def load_capacity_series(path=SOH_TABLE_PATH):
    """读取每只电池的逐圈充电容量序列，返回 {test_name: np.array(50,)}

    取容量表中 line==37（充电段）的记录：charging_capacity 为该圈真实充电
    容量（随 SEI 老化衰减），50 圈逐圈排列。
    """
    frame = pd.read_csv(path)
    frame = frame[frame.line == 37]
    cells = {}
    for name, group in frame.groupby('test_name'):
        group = group.sort_values('cycle_count')
        cells[name] = group['charging_capacity'].to_numpy(dtype=float)
    return cells


def parse_temperature(test_name):
    """从 test_name（如 000-SI-3.0-2500-S）解析温度℃（末段/100）"""
    return int(test_name.split('-')[-2]) / 100.0


def resample_curve(soh, scale):
    """时间轴缩放重采样：scale<1 衰减更快、scale>1 更慢；尾部线性外推防截断

    new_axis 为缩放后的源坐标（0 ~ (n-1)*scale）；超出原曲线范围的部分
    用末两点斜率线性外推，避免平线假象。
    """
    n = len(soh)
    axis = np.linspace(0.0, (n - 1) * scale, n)
    out = np.interp(axis, np.arange(n), soh)
    over = axis > n - 1
    if over.any():
        slope = soh[-1] - soh[-2]
        out[over] = soh[-1] + slope * (axis[over] - (n - 1))
    return out


def build_samples(cells):
    """构造 (X, Y, meta)：X=(N,20,1) 前20圈SOH，Y=(N,30) 后30圈SOH，meta=(电池名,缩放)"""
    x_list, y_list, meta = [], [], []
    for name, cap in cells.items():
        soh = cap / cap[0]                     # SOH 归一化（首圈 = 1.0）
        for scale in SCALES:
            curve = resample_curve(soh, scale)
            x_list.append(curve[:IN_LEN])
            y_list.append(curve[IN_LEN:IN_LEN + OUT_LEN])
            meta.append((name, scale))
    x = np.asarray(x_list, dtype=np.float32)[..., None]
    y = np.asarray(y_list, dtype=np.float32)
    return x, y, meta


def split_indices(cell_names_sorted, meta):
    """按电池分组划分 train/val/test 样本下标（增强样本跟随源电池）"""
    n_test = TEST_CELL_COUNT
    n_val = VAL_CELL_COUNT
    train_cells = set(cell_names_sorted[:-n_test - n_val])
    val_cells = set(cell_names_sorted[-n_test - n_val:-n_test])
    test_cells = set(cell_names_sorted[-n_test:])
    idx = {'train': [], 'val': [], 'test': []}
    for i, (name, _scale) in enumerate(meta):
        if name in train_cells:
            idx['train'].append(i)
        elif name in val_cells:
            idx['val'].append(i)
        elif name in test_cells:
            idx['test'].append(i)
    return idx, test_cells


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model():
    """小型 LSTM：序列编码 -> 直接回归未来 30 圈 SOH 曲线"""
    tf.random.set_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IN_LEN, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(OUT_LEN),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse', metrics=['mae'])
    return model


def predict_future_soh(model, soh_first20):
    """给定前 20 圈 SOH 序列，返回后 30 圈预测（已做单调化后处理）"""
    x = np.asarray(soh_first20, dtype=np.float32).reshape(1, IN_LEN, 1)
    pred = model.predict(x, verbose=0)[0]
    # 物理约束：容量衰减曲线应单调不增，累积最小值去除细微上翘
    return np.minimum.accumulate(pred)


# ---------------------------------------------------------------------------
# RUL：末端线性外推
# ---------------------------------------------------------------------------
def linear_extrapolate_rul(soh_curve, threshold=EOL_THRESHOLD,
                           tail_points=TAIL_FIT_POINTS):
    """末端线性外推求 EOL 循环数，返回 (eol_cycle_1based 或 None, slope)

    x 为 0-based 圈索引；eol_cycle 换算为 1-based 圈数（第 N 圈到达阈值）。
    """
    curve = np.asarray(soh_curve, dtype=float)
    n = len(curve)
    fit_len = min(tail_points, n)
    x = np.arange(n - fit_len, n, dtype=float)
    y = curve[n - fit_len:]
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= -1e-12:
        return None, float(slope)
    eol_cycle = (threshold - intercept) / slope
    return float(eol_cycle) + 1.0, float(slope)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate(model, cells, test_cells, meta, x_all, y_all):
    """测试集（真实曲线，scale=1.0）评估：曲线误差 + RUL 误差逐电池统计"""
    rows = []
    for i, (name, scale) in enumerate(meta):
        if scale != 1.0 or name not in test_cells:
            continue
        true_future = y_all[i]                                  # (30,)
        pred_future = predict_future_soh(model, x_all[i, :, 0])  # (30,)
        err = pred_future - true_future
        soh_true = cells[name] / cells[name][0]
        # 真实 RUL 与预测 RUL：20 圈已知 + 后段（真实/预测）曲线做末端外推
        true_eol, _ = linear_extrapolate_rul(soh_true)
        pred_eol, _ = linear_extrapolate_rul(
            np.concatenate([soh_true[:IN_LEN], pred_future]))
        rows.append({
            'test_name': name,
            'temperature_c': parse_temperature(name),
            'curve_mae': float(np.mean(np.abs(err))),
            'curve_rmse': float(np.sqrt(np.mean(err ** 2))),
            'final_err': float(err[-1]),
            'true_eol_cycle': true_eol,
            'pred_eol_cycle': pred_eol,
            'rul_err_cycle': (None if (true_eol is None or pred_eol is None)
                              else float(pred_eol - true_eol)),
        })
    summary = {
        'cells': rows,
        'curve_mae_mean': float(np.mean([r['curve_mae'] for r in rows])),
        'curve_rmse_mean': float(np.mean([r['curve_rmse'] for r in rows])),
        'final_err_mean': float(np.mean([r['final_err'] for r in rows])),
        'rul_err_abs_mean': float(np.mean([abs(r['rul_err_cycle']) for r in rows
                                           if r['rul_err_cycle'] is not None])),
    }
    return summary


# ---------------------------------------------------------------------------
# 演示图
# ---------------------------------------------------------------------------
def plot_results(history, model, cells, x_all, y_all, meta, idx, test_cells,
                 summary, path=RUL_PNG_PATH):
    """四联图：(a) 训练曲线 (b) 训练样例 (c) 测试泛化 (d) RUL 外推示意"""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    cycle_axis = np.arange(1, IN_LEN + OUT_LEN + 1)  # 1..50

    # (a) 训练/验证 loss
    ax = axes[0, 0]
    ax.plot(history.history['loss'], label='训练 loss', color='#1f77b4')
    ax.plot(history.history['val_loss'], label='验证 loss', color='#d62728')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE（log）')
    ax.set_title('(a) LSTM 训练曲线')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) 训练样例：前 3 只电池（scale=1.0）
    ax = axes[0, 1]
    first_name = meta[idx['train'][0]][0]
    for i in idx['train']:
        name, scale = meta[i]
        if scale != 1.0:
            continue
        soh_true = cells[name] / cells[name][0]
        pred = predict_future_soh(model, x_all[i, :, 0])
        ax.plot(cycle_axis[:IN_LEN], soh_true[:IN_LEN], color='#888888', lw=1.0,
                ls=':', label='已知前 20 圈' if name == first_name else None)
        ax.plot(cycle_axis[IN_LEN:], soh_true[IN_LEN:], color='black', lw=1.6,
                label='真实后 30 圈' if name == first_name else None)
        ax.plot(cycle_axis[IN_LEN:], pred, color='#d62728', lw=1.4, ls='--',
                label='LSTM 预测' if name == first_name else None)
        ax.text(cycle_axis[-1] + 0.5, soh_true[-1], name[:3], fontsize=8,
                va='center', color='black')
    ax.set_xlabel('循环数')
    ax.set_ylabel('SOH')
    ax.set_title('(b) 训练电池样例（000/001/002）')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)

    # (c) 测试泛化：后 4 只测试电池
    ax = axes[1, 0]
    test_sorted = sorted(test_cells)
    colors = ['#1f77b4', '#e377c2', '#2ca02c', '#9467bd', '#ff7f0e']
    for j, name in enumerate(test_sorted[-4:]):
        i = meta.index((name, 1.0))
        soh_true = cells[name] / cells[name][0]
        pred = predict_future_soh(model, x_all[i, :, 0])
        ax.plot(cycle_axis[IN_LEN:], soh_true[IN_LEN:], color=colors[j], lw=1.6,
                label=f'{name[:3]} 真实')
        ax.plot(cycle_axis[IN_LEN:], pred, color=colors[j], lw=1.2, ls='--',
                label=f'{name[:3]} 预测')
    ax.set_xlabel('循环数')
    ax.set_ylabel('SOH')
    ax.set_title('(c) 测试泛化（未见过的高温电池）')
    ax.legend(fontsize=7, ncol=2, loc='lower left')
    ax.grid(alpha=0.3)

    # (d) RUL 外推示意：最后一只测试电池（温度最高）
    ax = axes[1, 1]
    name = test_sorted[-1]
    i = meta.index((name, 1.0))
    soh_true = cells[name] / cells[name][0]
    pred = predict_future_soh(model, x_all[i, :, 0])
    full_pred = np.concatenate([soh_true[:IN_LEN], pred])
    pred_eol, slope = linear_extrapolate_rul(full_pred)
    true_eol, _ = linear_extrapolate_rul(soh_true)
    ax.plot(cycle_axis, soh_true, color='black', lw=1.6, label='真实 SOH（50 圈）')
    ax.plot(cycle_axis[IN_LEN:], pred, color='#d62728', lw=1.6, ls='--',
            label='LSTM 预测（后 30 圈）')
    ax.axhline(EOL_THRESHOLD, color='#2ca02c', lw=1.2, ls=':',
               label=f'EOL 阈值 SOH={EOL_THRESHOLD:.0%}')
    if pred_eol is not None:
        x_ext = np.array([cycle_axis[-1], pred_eol])
        y_ext = np.array([full_pred[-1], EOL_THRESHOLD])
        ax.plot(x_ext, y_ext, color='#d62728', lw=1.0, ls=':',
                label=f'线性外推（预测 RUL≈{pred_eol - IN_LEN:.0f} 圈）')
        ax.plot([pred_eol], [EOL_THRESHOLD], 'v', color='#d62728', ms=8)
    if true_eol is not None:
        ax.axvline(true_eol, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.text(true_eol, 0.86, f'真实 EOL≈{true_eol:.0f} 圈', rotation=90,
                fontsize=8, va='bottom', ha='right')
    ax.axvline(IN_LEN + 0.5, color='#888888', lw=0.8, alpha=0.6)
    ax.text(IN_LEN + 1.5, 1.005, '预测起点（第 20 圈）', fontsize=8,
            color='#555555')
    ax.set_xlabel('循环数')
    ax.set_ylabel('SOH')
    ax.set_ylim(min(0.75, full_pred.min() - 0.01), 1.01)
    ax.set_title(f'(d) RUL 外推示意 —— {name[:3]}（温度 {parse_temperature(name):.2f} ℃）')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)

    fig.suptitle('Step 3 · LSTM 容量衰减预测与 RUL 估算（Si 半电池数字孪生）',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[RUL] 演示图已保存：{path}')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    print('[RUL] 读取 SOH 数据集容量序列...')
    cells = load_capacity_series()
    cell_names = sorted(cells)
    print(f'[RUL] 电池 {len(cell_names)} 只，每只 {len(cells[cell_names[0]])} 圈')

    x_all, y_all, meta = build_samples(cells)
    idx, test_cells = split_indices(cell_names, meta)
    print(f'[RUL] 样本 {len(meta)} 个（{len(SCALES)} 倍增强）：'
          f'train {len(idx["train"])} / val {len(idx["val"])} / test {len(idx["test"])}')
    print(f'[RUL] 泛化测试电池（未见过，高温组）：'
          f'{", ".join(n[:3] for n in sorted(test_cells))}')

    model = build_model()
    model.summary(print_fn=lambda s: None)  # 静默结构输出，保持日志简洁
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60,
                                         restore_best_weights=True),
    ]
    history = model.fit(
        x_all[idx['train']], y_all[idx['train']],
        validation_data=(x_all[idx['val']], y_all[idx['val']]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=callbacks)
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    print(f'[RUL] 训练完成：{len(history.history["loss"])} epochs，'
          f'最佳 val_loss = {min(history.history["val_loss"]):.3e}（第 {best_epoch} epoch）')

    summary = evaluate(model, cells, test_cells, meta, x_all, y_all)
    print('[RUL] 测试集（10 只高温电池，真实曲线）：')
    print(f'      曲线 MAE  {summary["curve_mae_mean"]*100:.3f}% SOH（均值）')
    print(f'      曲线 RMSE {summary["curve_rmse_mean"]*100:.3f}% SOH（均值）')
    print(f'      末端误差  {summary["final_err_mean"]*100:+.3f}% SOH（均值）')
    print(f'      RUL 外推误差 {summary["rul_err_abs_mean"]:.1f} 圈（绝对值均值）')

    # 保存产物
    model.save(MODEL_PATH)
    pd.DataFrame(history.history).to_csv(HISTORY_CSV_PATH, index=False)
    with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f'[RUL] 模型已保存：{MODEL_PATH}')

    plot_results(history, model, cells, x_all, y_all, meta, idx, test_cells, summary)
    print('[RUL] 完成（Step 3/4）')


if __name__ == '__main__':
    main()
