# -*- coding: utf-8 -*-
"""Step 2：扩展卡尔曼滤波（EKF）在线 SOC 估算 —— 硅基负极数字孪生原型（第 2/4 步）。

功能
----
1. 读取 PyBaMM 统一仿真数据集（data/si-c-half-cell/，DFN(P2D)+热-力耦合，
   含 SEI 与裂纹老化）：OCV-SOC 先验表（由低倍率化成段生成）+ 指定电池/循环
   的放电段电压、电流序列；
2. 使用 filterpy 的扩展卡尔曼滤波（EKF），以一阶等效电路模型
   （OCV 查表 + 欧姆内阻 R0 + R1C1 极化）为过程模型，把仿真电压曲线当作
   "传感器观测"，逐采样点实时估算 SOC，把离线仿真变成在线估算；
3. 与纯安时积分（Coulomb Counting）对比，量化 EKF 对初始偏差与观测噪声的修正；
4. 独立运行生成本地展示图 results/digital_twin/ekf_soc_demo.png。

模型（放电为负电流约定，与数据集一致）
--------------------------------------
状态: x = [SOC, V1]，V1 为 R1C1 极化电压
预测: SOC(k+1) = SOC(k) + I·Δt/(Q·3600)
      V1(k+1)  = a·V1(k) + R1(1-a)·I,   a = exp(-Δt/(R1·C1))
观测: V = OCV(SOC) + R0·I + V1
参数: Q 为该段放电容量（Ah，由数据给出，随 SEI 老化衰减）；
      R0 由演示段网格辨识，R1/τ 为默认标定值（C/2 段极化由 RC 环节
      与 R0 共同吸收，电压观测负责闭环修正）。

用法
----
python kalman_soc.py            # 默认 000 号电池，输出 PNG 与指标
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免弹窗阻塞（app.py 同时复用本模块）
import matplotlib.pyplot as plt

from filterpy.kalman import ExtendedKalmanFilter

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SOC_DATA_DIR = os.path.join(REPO_ROOT, 'data', 'si-c-half-cell')
CYCLE_CSV_PATH = os.path.join(SOC_DATA_DIR, 'test_result.csv')
OCV_TABLE_PATH = os.path.join(SOC_DATA_DIR, 'si_c_half_cell_soc_train.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'digital_twin')
EKF_PNG_PATH = os.path.join(OUT_DIR, 'ekf_soc_demo.png')

DISCHARGE_LINE = 40        # 数据集中放电段的行号（UNIBO 约定）

# --- EKF 默认参数（可在 app 中调整）---
DEFAULT_SOC0 = 0.85        # 初始 SOC 猜测（故意偏离真值 1.0，展示 EKF 修正能力）
DEFAULT_SIGMA_V = 0.003    # 电压观测噪声标准差 [V]（模拟传感器噪声）
DEFAULT_R1 = 2.0           # 极化电阻 [Ω]（C/2 段极化压降较小）
DEFAULT_TAU_S = 120.0      # 极化时间常数 [s]
Q_SOC = 1e-8               # SOC 过程噪声方差
Q_V1 = 1e-6                # 极化电压过程噪声方差
P0_SOC = 0.01              # SOC 初始协方差
P0_V1 = 1e-4               # 极化电压初始协方差

# 中文字体（与数据生成脚本绘图风格一致）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_ocv_table(path=OCV_TABLE_PATH):
    """加载 OCV-SOC 先验表，返回 (soc_grid, ocv_grid)，soc ∈ [0,1]"""
    table = pd.read_csv(path)
    soc_grid = table['SOC [%]'].to_numpy(dtype=float) / 100.0
    ocv_grid = table['Voltage [V]'].to_numpy(dtype=float)
    return soc_grid, ocv_grid


def make_ocv_interp(soc_grid, ocv_grid):
    """由 OCV-SOC 表构造插值函数 ocv(s) 与导数 docv(s)（数值差分）

    入参 s 允许为任意形状数组（filterpy 状态向量元素），内部取标量返回。
    """

    def ocv(soc):
        soc = float(np.asarray(soc).reshape(-1)[0])
        return float(np.interp(soc, soc_grid, ocv_grid))

    def docv(soc):
        soc = float(np.asarray(soc).reshape(-1)[0])
        eps = 1e-4
        s_hi = min(soc + eps, 1.0)
        s_lo = max(soc - eps, 0.0)
        if s_hi == s_lo:
            return 0.0
        return float((np.interp(s_hi, soc_grid, ocv_grid)
                      - np.interp(s_lo, soc_grid, ocv_grid)) / (s_hi - s_lo))

    return ocv, docv


def load_discharge_segment(cell_name='000-SI-3.0-2500-S', cycle=3,
                           cycle_csv=CYCLE_CSV_PATH):
    """读取指定电池/循环的放电段（"传感器观测"数据）

    返回 dict: t[s], v[V], i[A]（放电为负）, soc_ref（参考 SOC，容量分数 1→0）,
    q_ah（段放电容量）, capacity_mah。
    """
    frame = pd.read_csv(cycle_csv)
    seg = frame[(frame.test_name == cell_name)
                & (frame.line == DISCHARGE_LINE)
                & (frame.cycle_count == cycle)].reset_index(drop=True)
    if len(seg) == 0:
        raise ValueError(f'未找到 {cell_name} cycle {cycle} 的放电段数据')
    # 数据集 step_time 单位为小时，统一换算为秒参与积分与滤波
    t = seg['step_time'].to_numpy(dtype=float) * 3600.0
    v = seg['voltage'].to_numpy(dtype=float)
    i = seg['current'].to_numpy(dtype=float)
    q = seg['discharging_capacity'].to_numpy(dtype=float)
    return {
        'cell_name': cell_name,
        'cycle': cycle,
        't': t,
        'v': v,
        'i': i,
        'soc_ref': 1.0 - q / q[-1],
        'q_ah': float(q[-1]),
        'capacity_mah': float(q[-1]) * 1000.0,
        'n_points': len(seg),
    }


# ---------------------------------------------------------------------------
# 参数辨识：对电压曲线网格搜索最优 R0
# ---------------------------------------------------------------------------
def calibrate_r0(segment, ocv, r1=DEFAULT_R1, tau=DEFAULT_TAU_S,
                 candidates=None):
    """用参考 SOC 轨迹 + 观测电压，网格搜索使端电压拟合误差最小的 R0

    开环预测电压：V_pred = OCV(soc_ref) + R0·I + V1（V1 由 RC 递推），
    返回 (best_r0, 误差表) 供演示图绘制。
    """
    if candidates is None:
        candidates = np.arange(0.0, 30.5, 0.5)
    t = segment['t']
    i = segment['i']
    v = segment['v']
    soc_ref = segment['soc_ref']
    errors = []
    for r0 in candidates:
        v1 = 0.0
        v_pred = np.empty_like(v)
        for k in range(len(t)):
            dt = t[k] - t[k - 1] if k > 0 else 0.0
            a = np.exp(-dt / tau) if dt > 0 else 1.0
            v1 = a * v1 + r1 * (1.0 - a) * i[k]
            v_pred[k] = ocv(soc_ref[k]) + r0 * i[k] + v1
        errors.append(np.sqrt(np.mean((v_pred - v) ** 2)))
    errors = np.asarray(errors)
    best_index = int(np.argmin(errors))
    return float(candidates[best_index]), {'candidates': candidates, 'rmse': errors}


# ---------------------------------------------------------------------------
# EKF 与对比方法
# ---------------------------------------------------------------------------
def run_ekf(segment, ocv, docv, r0, soc0=DEFAULT_SOC0, sigma_v=DEFAULT_SIGMA_V,
            r1=DEFAULT_R1, tau=DEFAULT_TAU_S, v_obs=None):
    """逐采样点运行 filterpy 扩展卡尔曼滤波，返回 SOC 估计与预测电压

    v_obs 缺省时使用段原始电压（无噪声）。
    """
    t, i = segment['t'], segment['i']
    q_ah = segment['q_ah']
    v_obs = segment['v'] if v_obs is None else np.asarray(v_obs, dtype=float)

    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.x = np.array([[soc0], [0.0]])
    ekf.P = np.diag([P0_SOC, P0_V1])
    ekf.R = np.array([[sigma_v ** 2]])
    ekf.Q = np.diag([Q_SOC, Q_V1])
    # 注：filterpy 的 EKF 不使用 self.H 属性（update 时传入 h_jacobian），
    # 也没有 self.u 属性——控制输入必须经 predict(u=...) 显式传入。

    def hx(x, current):
        return np.array([ocv(x[0]) + r0 * current + x[1]])

    def h_jacobian(x, current):
        return np.array([[docv(x[0]), 1.0]])

    soc_est = np.empty(len(t))
    v_pred = np.empty(len(t))
    for k in range(len(t)):
        dt = t[k] - t[k - 1] if k > 0 else 0.0
        a = np.exp(-dt / tau) if dt > 0 else 1.0

        # --- 预测：SOC 安时积分 + 极化 RC 递推 ---
        # 电流项 B·u 必须经 predict(u=...) 传入：filterpy 的 predict() 默认
        # u=0 且不读取任何 self.u 属性，漏传时电流驱动项被静默置零，
        # SOC 预测不前进（只能靠电压观测硬拉，估计严重滞后）。
        ekf.F = np.array([[1.0, 0.0], [0.0, a]])
        ekf.B = np.array([[dt / (q_ah * 3600.0)], [r1 * (1.0 - a)]])
        ekf.predict(u=np.array([[i[k]]]))

        # --- 更新：端电压观测（SOC 物理约束裁剪到 [0,1]）---
        ekf.x[0, 0] = min(max(ekf.x[0, 0], 0.0), 1.0)
        ekf.update(np.array([v_obs[k]]), h_jacobian, hx,
                   args=(i[k],), hx_args=(i[k],))

        soc_est[k] = ekf.x[0, 0]
        v_pred[k] = ocv(ekf.x[0, 0]) + r0 * i[k] + ekf.x[1, 0]

    return {'soc': np.clip(soc_est, 0.0, 1.0), 'v_pred': v_pred}


def run_coulomb_counting(segment, soc0=DEFAULT_SOC0):
    """纯安时积分（不修正）：SOC(k) = soc0 + Σ I·Δt/(Q·3600)"""
    t, i = segment['t'], segment['i']
    q_ah = segment['q_ah']
    soc = np.empty(len(t))
    acc = 0.0
    soc[0] = soc0
    for k in range(1, len(t)):
        acc += i[k] * (t[k] - t[k - 1])
        soc[k] = soc0 + acc / (q_ah * 3600.0)
    return np.clip(soc, 0.0, 1.0)


def compute_metrics(soc_ref, soc_est):
    """SOC 估计指标：RMSE / MAE / 末点误差（均为 SOC 绝对量，0~1）"""
    err = np.asarray(soc_est) - np.asarray(soc_ref)
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'final_err': float(err[-1]),
    }


# ---------------------------------------------------------------------------
# 演示：000 号电池整流程 + 出图
# ---------------------------------------------------------------------------
def run_demo(cell_name='000-SI-3.0-2500-S', cycle=3, soc0=DEFAULT_SOC0,
             sigma_v=DEFAULT_SIGMA_V, r0=None, noise_seed=42,
             add_noise=True, save_png=True):
    """运行 EKF 在线 SOC 估算演示，返回结果字典（供 app 复用）"""
    soc_grid, ocv_grid = load_ocv_table()
    ocv, docv = make_ocv_interp(soc_grid, ocv_grid)
    segment = load_discharge_segment(cell_name, cycle)

    # R0 辨识（未指定时网格搜索）
    r0_cal, cal_table = calibrate_r0(segment, ocv)
    r0 = r0_cal if r0 is None else float(r0)

    # 观测电压（模拟传感器：真实曲线 + 高斯噪声）
    rng = np.random.default_rng(noise_seed)
    v_obs = segment['v'] + (rng.normal(0.0, sigma_v, len(segment['v']))
                            if add_noise else 0.0)

    ekf_result = run_ekf(segment, ocv, docv, r0, soc0=soc0,
                         sigma_v=sigma_v, v_obs=v_obs)
    soc_cc = run_coulomb_counting(segment, soc0=soc0)

    metrics_ekf = compute_metrics(segment['soc_ref'], ekf_result['soc'])
    metrics_cc = compute_metrics(segment['soc_ref'], soc_cc)

    result = {
        'segment': segment,
        'soc_ref': segment['soc_ref'],
        'soc_ekf': ekf_result['soc'],
        'soc_cc': soc_cc,
        'v_obs': v_obs,
        'r0': r0,
        'metrics_ekf': metrics_ekf,
        'metrics_cc': metrics_cc,
        'calibration': cal_table,
    }

    print(f'[EKF] 电池 {cell_name} cycle {cycle}：{segment["n_points"]} 个采样点，'
          f'段容量 {segment["capacity_mah"]:.4f} mAh')
    print(f'[EKF] R0 标定值 {r0:.1f} Ω；初始 SOC 猜测 {soc0:.2f}（真值 1.00）')
    print(f'[EKF] EKF   RMSE {metrics_ekf["rmse"]*100:.2f}% ，末点误差 '
          f'{metrics_ekf["final_err"]*100:+.2f}%')
    print(f'[EKF] 安时积分 RMSE {metrics_cc["rmse"]*100:.2f}% ，末点误差 '
          f'{metrics_cc["final_err"]*100:+.2f}%')

    if save_png:
        plot_demo(result, path=EKF_PNG_PATH)
    return result


def plot_demo(result, path=None):
    """绘制三步演示图：观测电压 / SOC 对比 / SOC 误差"""
    segment = result['segment']
    t_min = segment['t'] / 60.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) 电压观测
    axes[0].plot(t_min, segment['v'], lw=1.0, color='#888888', label='仿真电压（原始）')
    axes[0].plot(t_min, result['v_obs'], lw=0.8, color='#1f77b4', alpha=0.8,
                 label='传感器观测（含噪声）')
    axes[0].set_xlabel('时间 [min]')
    axes[0].set_ylabel('端电压 [V]')
    axes[0].set_title('(a) 电压观测（传感器）')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].grid(alpha=0.3)

    # (b) SOC 对比
    axes[1].plot(segment['soc_ref'] * 100.0, t_min, color='black', lw=2.0,
                 label='参考 SOC（容量分数）')
    axes[1].plot(result['soc_ekf'] * 100.0, t_min, color='#d62728', lw=1.8,
                 label='EKF 在线估计')
    axes[1].plot(result['soc_cc'] * 100.0, t_min, color='#1f77b4', lw=1.2,
                 ls='--', label=f'安时积分（初值 {DEFAULT_SOC0*100:.0f}%）')
    axes[1].set_xlabel('SOC [%]')
    axes[1].set_ylabel('时间 [min]')
    axes[1].set_title('(b) SOC 估计：EKF vs 安时积分')
    axes[1].legend(fontsize=8, loc='best')
    axes[1].grid(alpha=0.3)
    axes[1].invert_yaxis()  # 时间向下：按真实时间流方向看 SOC 变化

    # (c) SOC 误差
    err_ekf = (result['soc_ekf'] - segment['soc_ref']) * 100.0
    err_cc = (result['soc_cc'] - segment['soc_ref']) * 100.0
    axes[2].plot(t_min, err_ekf, color='#d62728', lw=1.6,
                 label=f'EKF（RMSE {result["metrics_ekf"]["rmse"]*100:.2f}%）')
    axes[2].plot(t_min, err_cc, color='#1f77b4', lw=1.2, ls='--',
                 label=f'安时积分（RMSE {result["metrics_cc"]["rmse"]*100:.2f}%）')
    axes[2].axhline(0.0, color='black', lw=0.8, alpha=0.5)
    axes[2].set_xlabel('时间 [min]')
    axes[2].set_ylabel('SOC 误差 [%]')
    axes[2].set_title('(c) SOC 误差（滤波修正 vs 纯积分漂移）')
    axes[2].legend(fontsize=8, loc='best')
    axes[2].grid(alpha=0.3)

    fig.suptitle(f'Step 2 · EKF 在线 SOC 估算 —— {segment["cell_name"]} '
                 f'cycle {segment["cycle"]}（C/2 循环，含 SEI 老化）', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f'[EKF] 演示图已保存：{path}')
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_demo()
    print('[EKF] 完成（Step 2/4）')


if __name__ == '__main__':
    main()
