# -*- coding: utf-8 -*-
"""EIS 阻抗仿真脚本：小信号正弦扫频 + 标准 Nyquist 图 + 等效电路参数提取。

功能
----
1. 在 DFN（P2D）半电池模型上做**时域小信号正弦激励**（恒流 galvanostatic EIS）：
   对每个 (温度, SOC) 工作点，先在目标 SOC 的平衡态初始化（把正极初始浓度设为
   目标 SOC 对应值），再施加 I(t) = A·sin(2πft) 的小振幅正弦电流（默认 C/200，
   响应约 mV 级，满足线性小信号条件）；
2. 关键模型设置：``'surface form': 'differential'``——PyBaMM 的双电层容量
   （Cdl）只在 surface form（表面电位形式电解质电导率）子模型中进入方程；
   默认 'false' 时双电层不生效，谱会缺失中频 RC 半圆弧（表现为纯电阻平台）。
   开启后得到教科书级的「高频截距 → 中频半圆（Rct‖CPE）→ 低频扩散尾」谱形；
3. 模型为**等温**：'surface form' 与 'thermal': 'lumped' 存在库级 ShapeError
   （状态维度不匹配）；小信号短时测试下温升 <0.05 K，等温假设合理；
4. 输出：
   * ``results/eis/eis_impedance.csv``：长表（温度, SOC, 频率, Z', Z'', |Z|, 相位）；
   * ``results/eis/eis_nyquist.png``：**等比例坐标**标准 Nyquist 图（含 Randles
     等效电路示意图与 R0/弧/扩散尾标注，按温度分面、SOC 分色）；
   * ``results/eis/eis_fit_params.csv`` + ``eis_fit_trends.png``：
     ZARC 等效电路拟合（R0 + Rct‖CPE + 半无限 Warburg）的参数提取——高频
     截距→欧姆内阻 R0、中频半圆→电荷转移 Rct（+CPE 参数 Q/n）、低频斜线→
     Warburg 系数 σ，并给出 R0/Rct 随 SOC/温度的演变趋势；
5. 典型物理结论（可直接用于报告/面试）：高频截距随温度升高而下降（电导率与
   动力学 Arrhenius 加速）、半圆直径随 SOC 变化（交换电流密度 j0 变化）、
   低温/低 SOC 下扩散尾更显著（固相扩散受限）。

方法学说明（时域小信号 G-EIS 的实测要点）
------------------------------------------
- 小信号条件：响应幅度控制在 5~15 mV；默认振幅 C/200（≈0.06 mA）；
- 瞬态衰减：每频点仿真时长取 max(最小周期数×周期, MIN_WARMUP_S)，覆盖
  电荷转移时间常数（Rct·Cdl ≈ 0.02 s）的多个时间常数；
- 频点链式衔接：从高频到低频逐点扫，每点以上一频点终态为初值，抑制
  各频点独立重启导致的曲线折返；
- 基波提取必须用**最小二乘拟合**（DC+cos+sin）：端电压含 ~0.3~0.6 V 直流
  分量，是 mV 级响应的几十倍，相关法/DFT 在含重复端点的窗上会残留
  DC×2/N 泄漏，彻底污染结果；
- 低频局限：固相扩散时间常数 R²/D ≈ 2000 s，<0.1 Hz 为“过渡区近似”
  （趋势可信、绝对值偏保守），如需更准扩散尾可增大 LOW_FREQ_PERIODS。

计算量与配置
------------
- 默认 4 个代表温度（25/30/35/40 ℃）× 5 个 SOC（90/70/50/30/10%）× 13 个
  频率对数点（1000 Hz ~ 0.05 Hz，覆盖半圆弧与扩散尾）；
  全温度网格（31 档）请把 TEMPERATURES_C 改为 None（计算量 ×7）；
- 环境变量 ``EIS_QUICK=1`` 冒烟模式：1 个温度 × 4 个 SOC × 8 个频点；
- 环境变量 ``EIS_REFIT=1``：不重新仿真，直接读 eis_impedance.csv 重拟合参数；
- 仿真驱动优先用 InputParameter 传频率（一次构建多次求解）；若当前 PyBaMM
  版本不支持（构建/求解报错），自动切换为「每频点重建」的兼容路径。

用法
----
python eis_simulation.py          # 完整扫频（默认配置，CPU 数十分钟级）
（可选前置：先运行 si_halfcell_dataset.py 生成 OCV 表，使 SOC→化学计量比
 映射更精确；无表时自动退化为线性近似。）
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面后端，避免弹窗阻塞
import matplotlib.pyplot as plt
import pybamm
from scipy.optimize import least_squares

from si_halfcell_dataset import (
    C_MAX, NOMINAL_CAPACITY, STO_INIT, TEMP_START_C, TEMP_STEP_C, TEMP_STOP_C,
    VAR_PTS_OVERRIDE, X_FULL, _base_parameter_values,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'eis')
IMPEDANCE_CSV_PATH = os.path.join(OUT_DIR, 'eis_impedance.csv')
FIT_CSV_PATH = os.path.join(OUT_DIR, 'eis_fit_params.csv')
OCV_TABLE_PATH = os.path.join(REPO_ROOT, 'data', 'si-c-half-cell',
                              'si_c_half_cell_soc_train.csv')

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
EIS_QUICK = os.environ.get('EIS_QUICK', '').strip() not in ('', '0', 'false', 'False')
# EIS_REFIT=1：不重新仿真，直接读 eis_impedance.csv 重拟合等效电路参数
# （调整拟合模型/边界后快速刷新拟合结果）
EIS_REFIT = os.environ.get('EIS_REFIT', '').strip() not in ('', '0', 'false', 'False')


def _all_temperatures():
    """数据集全温度网格（25.0 ~ 40.0 ℃，步长 0.5 ℃，共 31 档）"""
    count = int(round((TEMP_STOP_C - TEMP_START_C) / TEMP_STEP_C)) + 1
    return [round(TEMP_START_C + TEMP_STEP_C * index, 2) for index in range(count)]


ALL_TEMPERATURES = _all_temperatures()

if EIS_QUICK:
    TEMPERATURES_C = [30.0]                              # 冒烟：单温度
    SOC_LEVELS = [0.9, 0.5, 0.3, 0.1]                    # 冒烟：四个 SOC
    # 冒烟：覆盖弧顶（30 Hz）与扩散尾的人工选点
    FREQUENCIES_HZ = np.array([1000.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1])
else:
    # 代表温度（4 档）；如需全温度网格改为：TEMPERATURES_C = None
    TEMPERATURES_C = [25.0, 30.0, 35.0, 40.0]
    SOC_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.1]
    # 1000 ~ 0.05 Hz：完整覆盖高频截距 → 半圆弧（~30 Hz 弧顶）→ 扩散尾
    FREQUENCIES_HZ = np.logspace(3, -1.3, 13)

AMPLITUDE_C_RATE = 0.005      # 激励振幅（C/200）：响应约 mV 级，线性小信号区
PERIODS_PER_FREQ = 4          # 常规频点仿真周期数
LOW_FREQ_THRESHOLD_HZ = 0.5   # 低频阈值（低于则加深暖机；高频段已验证平滑）
LOW_FREQ_PERIODS = 12         # 低频段周期数（扩散瞬态近似：τ_diff = R²/D ≈ 2000 s）
MIN_WARMUP_S = 0.15           # 每频点最小仿真时长 [s]（≥5×(Rct·Cdl)，覆盖电荷转移瞬态）
POINTS_PER_PERIOD = 32        # 每周期输出采样点数（相关检测足够）
ANALYSIS_PERIODS = 2          # 取最后 N 个完整周期做基波相量提取

# 模型选项（EIS 专用）：SEI 老化 + surface form（双电层动力学生效）。
# 要点：
# - PyBaMM 的 Cdl 只在 surface form（表面电位形式电解质电导率）子模型中进入
#   方程；默认 'surface form': 'false' 时双电层不生效，谱缺失中频 RC 半圆
#   （表现为纯电阻平台）。'differential' 与 'algebraic' 均可，前者保留 η 动态；
# - 'surface form' 与 'thermal': 'lumped' 组合存在库级 ShapeError（状态维度
#   不匹配），故 EIS 用等温——小信号短时测试温升 <0.05 K，等温合理；
# - 力学/裂纹状态在若干正弦周期内不变，对阻抗无贡献，不启用；
# - halfcell 参数集缺少锂金属对电极/负集流器热物性，_base_parameter_values
#   已代补（见 si_halfcell_dataset.py）。
EIS_MODEL_OPTIONS = {
    'working electrode': 'positive',
    'SEI': 'solvent-diffusion limited',
    'SEI film resistance': 'distributed',
    'SEI porosity change': 'true',
    'surface form': 'differential',
}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------------
# 工作点构建：SOC -> 初始浓度、正弦激励、仿真器
# ---------------------------------------------------------------------------
def target_concentration(soc):
    """目标 SOC(0~1) -> 正极初始锂浓度 [mol.m-3]

    优先读数据集 OCV 先验表（SOC-化学计量比映射，来自低倍率化成段）；
    缺表时退化为线性近似（仅供快速冒烟）。
    """
    if os.path.exists(OCV_TABLE_PATH):
        table = pd.read_csv(OCV_TABLE_PATH)
        x_target = float(np.interp(
            soc,
            table['SOC [%]'].to_numpy(dtype=float) / 100.0,
            table['Stoichiometry x'].to_numpy(dtype=float)))
    else:
        x_target = STO_INIT * X_FULL + (1.0 - soc) * (X_FULL - STO_INIT * X_FULL)
    x_target = float(np.clip(x_target, 1e-3, X_FULL * (1.0 - 1e-3)))
    return C_MAX * x_target / X_FULL


def _eis_current_with_inputs(t):
    """电流函数（InputParameter 驱动）：频率/振幅经 solve(inputs=...) 传入，
    使一个 (T, SOC) 工作点只需构建一次 Simulation 即可扫描全部频点。"""
    frequency = pybamm.InputParameter('EIS frequency [Hz]')
    amplitude = pybamm.InputParameter('EIS amplitude [A]')
    return amplitude * pybamm.sin(2.0 * np.pi * frequency * t)


def _fixed_frequency_current(frequency_hz, amplitude_a):
    """电流函数（固定频率）：兼容路径，每频点重建一次仿真"""
    def current(t):
        return amplitude_a * pybamm.sin(2.0 * np.pi * frequency_hz * t)
    return current


def _make_eis_solver():
    """EIS 只需端电压输出：优先限定输出变量（IDAKLU，显著提速），逐级兜底"""
    factories = (
        lambda: pybamm.IDAKLUSolver(output_variables=['Voltage [V]']),
        lambda: pybamm.IDAKLUSolver(),
    )
    for factory in factories:
        try:
            return factory()
        except Exception:  # noqa: BLE001 - 版本差异：尝试下一种构建方式
            continue
    return pybamm.CasadiSolver(mode='safe')


def build_point_simulation(temperature_c, soc, frequency_hz=None, amplitude_a=None):
    """构建一个 (T, SOC) 工作点的 DFN 仿真

    frequency_hz 为 None 时使用 InputParameter 驱动（一次构建多次求解）；
    否则电流函数为固定频率正弦（兜底路径，每个频点调用一次）。
    """
    model = pybamm.lithium_ion.DFN(EIS_MODEL_OPTIONS)
    parameters = _base_parameter_values(273.15 + temperature_c)
    parameters.update({
        'Initial concentration in positive electrode [mol.m-3]':
            target_concentration(soc),
        'Current function [A]': (_eis_current_with_inputs if frequency_hz is None
                                 else _fixed_frequency_current(frequency_hz,
                                                               amplitude_a)),
    })
    return pybamm.Simulation(model, parameter_values=parameters,
                             # var_pts 需与默认网格合并（PyBaMM 26.x 整体覆盖语义）
                             var_pts={**model.default_var_pts, **VAR_PTS_OVERRIDE},
                             solver=_make_eis_solver())


# ---------------------------------------------------------------------------
# 单频点求解与阻抗提取
# ---------------------------------------------------------------------------
def _solve_and_extract(simulation, frequency_hz, amplitude_a, use_inputs,
                       initial_solution=None):
    """求解一个频点的正弦仿真并提取阻抗 Z(f)（复数），返回 (Z, solution)

    - 仿真时长取 max(最小周期数×周期, MIN_WARMUP_S)：高频点强制覆盖电荷转移
      瞬态时标，低频点用更多周期近似扩散瞬态；
    - initial_solution 非空时用 model.set_initial_conditions_from 做链式衔接
      （注意：PyBaMM 26.x 的 Simulation.solve 已移除 initial_conditions 参数，
      使用旧写法会被静默回退为独立重启，链路不生效）；
    - 基波提取用**最小二乘拟合**（DC + cos + sin 三基函数）：半电池端电压含
      ~0.57 V 直流分量（是 mV 级交流响应的几十倍），相关法/DFT 在含重复端点
      的采样窗上会残留 DC×2/N 的泄漏（实测 ~17 mV），彻底污染结果（表现为
      负虚部、孤立尖峰、伪相位）；拟合把 DC 一并解出，对采样非均匀也稳健。
    """
    period_s = 1.0 / frequency_hz
    base_periods = (LOW_FREQ_PERIODS if frequency_hz < LOW_FREQ_THRESHOLD_HZ
                    else PERIODS_PER_FREQ)
    n_periods = max(base_periods, int(np.ceil(MIN_WARMUP_S / period_s)))
    t_end = n_periods * period_s
    t_eval = np.linspace(0.0, t_end, n_periods * POINTS_PER_PERIOD + 1)

    # 链式衔接：用上一频点终态更新模型初值（失败则退化为独立重启）
    if initial_solution is not None:
        try:
            simulation.model.set_initial_conditions_from(initial_solution)
        except Exception as error:  # noqa: BLE001 - 衔接失败不中断扫频
            print(f'[提醒] 链式衔接失败（{error!r}），该频点独立重启')

    if use_inputs:
        solution = simulation.solve(t_eval=t_eval, inputs={
            'EIS frequency [Hz]': frequency_hz,
            'EIS amplitude [A]': amplitude_a,
        })
    else:
        solution = simulation.solve(t_eval=t_eval)

    # 均匀网格上的电压（插值）与解析电流（与模型时间轴同源，严格对齐相位）
    voltage = np.asarray(solution['Voltage [V]'](t_eval), dtype=float).ravel()
    t = t_eval

    # 取最后 ANALYSIS_PERIODS 个完整周期做最小二乘基波提取（DC+cos+sin）
    window_s = ANALYSIS_PERIODS * period_s
    mask = t >= (t_end - window_s) - 1e-9
    if int(mask.sum()) < 8:
        raise RuntimeError(f'分析窗采样点不足（{int(mask.sum())} 点）')
    t_window = t[mask]
    omega = 2.0 * np.pi * frequency_hz
    design = np.column_stack([np.ones_like(t_window),
                              np.cos(omega * t_window),
                              np.sin(omega * t_window)])
    coefficients, *_ = np.linalg.lstsq(design, voltage[mask], rcond=None)
    _, cos_coef, sin_coef = coefficients
    v_phasor = cos_coef - 1.0j * sin_coef        # V(t) 中 e^{jωt} 分量
    i_phasor = -1.0j * amplitude_a               # I(t) = A·sin(ωt) 的相量
    if not np.isfinite(v_phasor) or abs(i_phasor) < 1e-15:
        raise RuntimeError('端电压响应无效（NaN/零电流相量）')
    # 符号修正：PyBaMM 半电池电流约定“放电为正”，施加放电方向正电流使端电压
    # 下降（dV/dI < 0），与 EIS 惯例（阳极方向为正，Z 实部 > 0）相反；
    # 取负号得到常规第一象限 Nyquist 谱。
    return -(v_phasor / i_phasor), solution


def run_eis_job(temperature_c, soc, frequencies, amplitude_a):
    """跑一个 (T, SOC) 工作点的全频段扫频，返回与 frequencies 对齐的 Z 复数数组

    从最高频到最低频逐点扫描，频点间链式衔接（上一频点终态作为下一频点初值），
    抑制各频点独立重启导致的瞬态污染（曲线折返）；求解失败的频点写
    complex(NaN, NaN)，不中断整个扫描。优先 InputParameter 驱动（一次构建）；
    构建失败自动切换为每频点重建的兼容路径（兼容路径不做链式衔接）。
    """
    impedances = np.full(len(frequencies), complex(np.nan, np.nan), dtype=complex)
    use_inputs = True
    simulation = None
    state_solution = None
    for index, frequency in enumerate(frequencies):
        if use_inputs and simulation is None:
            try:
                simulation = build_point_simulation(temperature_c, soc)
            except Exception as error:  # noqa: BLE001 - 驱动方式不支持时切换
                print(f'[警告] InputParameter 驱动构建失败（{error!r}），'
                      f'切换为每频点重建模式')
                use_inputs = False
        try:
            if use_inputs:
                z, state_solution = _solve_and_extract(
                    simulation, frequency, amplitude_a, use_inputs=True,
                    initial_solution=state_solution)
            else:
                # 兼容路径：每频点使用独立构建的仿真（不跨参数集链式衔接）
                simulation = build_point_simulation(temperature_c, soc,
                                                    frequency, amplitude_a)
                z, _ = _solve_and_extract(
                    simulation, frequency, amplitude_a, use_inputs=False)
            impedances[index] = z
        except Exception as error:  # noqa: BLE001 - 单频点失败写 NaN 继续
            print(f'[提醒] {temperature_c:.1f} ℃ / SOC {soc:.0%} / '
                  f'{frequency:.4g} Hz 求解失败: {error!r}')
            state_solution = None  # 断裂后下一频点独立重启
    return impedances


# ---------------------------------------------------------------------------
# 等效电路拟合：ZARC（R0 + Rct‖CPE + 半无限 Warburg）
# ---------------------------------------------------------------------------
def _zarc_impedance(frequencies, parameters):
    """ZARC 等效电路阻抗：R0 + Rct/(1+Rct·Q(jω)^n) + σ/√(jω)

    Rct‖CPE 为电荷转移弧的通用形式（CPE 即双电层/界面容性，n→1 时
    退化为理想电容 C=Q）；尾部为半无限 Warburg 扩散。
    """
    r0, rct, q_cpe, n_cpe, sigma = parameters
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=float)
    z_rc = rct / (1.0 + rct * q_cpe * (1.0j * omega) ** n_cpe)
    z_w = sigma / np.sqrt(1.0j * omega)
    return r0 + z_rc + z_w


def fit_zarc(frequencies, impedance):
    """最小二乘拟合 ZARC 参数，返回 dict（含 rmse）或 None

    高频截距 → R0（欧姆内阻）；中频弧 → Rct（电荷转移 + SEI 膜阻）
    与 CPE（Q, n）；低频 45° 斜线 → σ（Warburg 扩散系数，需低于 ~0.1 Hz
    的频点才可辨识）。初值取数据特征值，避免差初值收敛到无物理意义的
    局部解（实测 Randles 初值 Cdl=1 F 会收敛失败，ZARC 形式更鲁棒）。
    """
    frequencies = np.asarray(frequencies, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)
    if len(frequencies) < 6:
        return None

    x0 = [max(float(np.min(impedance.real)), 1e-3),
          max(float(np.max(impedance.real) - np.min(impedance.real)), 1e-3),
          1e-3, 0.8,
          max(float(np.min(np.abs(impedance.imag))), 1e-3)]

    def residual(parameters):
        diff = _zarc_impedance(frequencies, parameters) - impedance
        return np.concatenate([diff.real, diff.imag])

    try:
        result = least_squares(
            residual, x0=x0,
            # 物理合理边界：R0/Rct ≤ 1e4 Ω、Q ≤ 1e2、n ∈ [0.3, 1]、σ ≤ 1e3
            bounds=([0.0, 0.0, 1e-9, 0.3, 0.0], [1e4, 1e4, 1e2, 1.0, 1e3]),
            max_nfev=8000)
    except Exception:  # noqa: BLE001 - 拟合失败返回 None
        return None
    r0, rct, q_cpe, n_cpe, sigma = (float(value) for value in result.x)
    rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
    return {'r0_ohm': r0, 'rct_ohm': rct, 'cpe_q': q_cpe, 'cpe_n': n_cpe,
            'sigma_warburg': sigma, 'rmse_ohm': rmse}


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def _draw_randles_circuit(ax, x0=0.46, y0=0.80):
    """在 Nyquist 图内绘制 Randles 等效电路示意（轴相对坐标）

    结构：—[R₀]—┬—[R_ct]—┬—[Z_w]—
                 └─║ C_dl ║─┘
    """
    from matplotlib.patches import Rectangle
    style = dict(transform=ax.transAxes, color='0.35', linewidth=1.0,
                 clip_on=False)
    box_w, box_h, wire, half = 0.075, 0.028, 0.026, 0.022
    y = y0

    def box(x_start, yy, label, fontsize=7):
        ax.add_patch(Rectangle((x_start, yy - box_h / 2), box_w, box_h,
                               fill=False, edgecolor='0.35', linewidth=1.0,
                               transform=ax.transAxes, clip_on=False))
        ax.text(x_start + box_w / 2, yy, label, transform=ax.transAxes,
                fontsize=fontsize, ha='center', va='center', color='0.2')

    x = x0
    ax.plot([x, x + wire], [y, y], **style)
    x += wire
    box(x, y, 'R₀')
    x += box_w
    ax.plot([x, x + wire], [y, y], **style)
    x += wire
    left = x
    ax.plot([left, left], [y - half, y + half], **style)
    x += 0.012
    # 上支路 R_ct
    ax.plot([x - 0.012, x + 0.010], [y + half, y + half], **style)
    box(x + 0.010, y + half, 'R_ct', fontsize=6.5)
    x2 = x + 0.010 + box_w
    # 下支路 C_dl（两条平行短线）
    c_mid = x + 0.010 + box_w * 0.5
    ax.plot([x - 0.012, c_mid - 0.008], [y - half, y - half], **style)
    ax.plot([c_mid - 0.008, c_mid - 0.008],
            [y - half - 0.016, y - half + 0.016], **style)
    ax.plot([c_mid + 0.008, c_mid + 0.008],
            [y - half - 0.016, y - half + 0.016], **style)
    ax.plot([c_mid + 0.008, x2], [y - half, y - half], **style)
    ax.text(c_mid, y - half - 0.030, 'C_dl', transform=ax.transAxes,
            fontsize=6.5, ha='center', va='top', color='0.2')
    # 右节点
    ax.plot([x2, x2], [y - half, y + half], **style)
    ax.text(x2 + 0.010, y + half, 'Z_w', transform=ax.transAxes,
            fontsize=6.5, ha='left', va='center', color='0.2')
    ax.plot([x2, x2 + wire], [y, y], **style)
    ax.text(x0, y + half + 0.026, 'Randles 等效电路', transform=ax.transAxes,
            fontsize=7, color='0.2', ha='left')


def plot_nyquist(records, out_png, temperatures):
    """标准 Nyquist 图：等比例坐标、按温度分面、SOC 分色

    - x 轴 = Z'（实部）、y 轴 = -Z''（负虚部），**等比例**（同尺度，标准画法）；
    - 可见高频截距（R0）→ 中频半圆弧（Rct‖Cdl）→ 低频扩散尾的完整谱形；
    - 图内附 Randles 等效电路示意；低频（<0.5 Hz）为过渡区近似，
      用虚线+空心点区分。
    """
    n_temp = len(temperatures)
    ncols = min(2, n_temp)
    nrows = int(np.ceil(n_temp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.2 * nrows),
                             squeeze=False, dpi=150)
    for index, temperature in enumerate(temperatures):
        ax = axes[index // ncols][index % ncols]
        subset = records[records['temperature_c'] == temperature]
        for soc, group in subset.groupby('soc'):
            group = group.sort_values('frequency_hz', ascending=False)
            reliable = group[group['frequency_hz'] >= LOW_FREQ_THRESHOLD_HZ]
            # 过渡区与可靠区共享边界点，保证虚线从实线末端接续（不断开）
            transitional = pd.concat([
                reliable.tail(1),
                group[group['frequency_hz'] < LOW_FREQ_THRESHOLD_HZ]])
            ax.plot(reliable['z_real_ohm'], -reliable['z_imag_ohm'], 'o-',
                    markersize=3.8, linewidth=1.3, label=f'SOC {soc:.0%}')
            ax.plot(transitional['z_real_ohm'], -transitional['z_imag_ohm'],
                    'o--', markersize=3.8, linewidth=1.2, mfc='white')
        ax.set_xlabel("Z' [Ω]")
        ax.set_ylabel("-Z'' [Ω]")
        ax.set_title(f'{temperature:.1f} ℃')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc='lower right')
        ax.set_aspect('equal', adjustable='datalim')   # 等比例：标准 Nyquist
        _draw_randles_circuit(ax)
        ax.text(0.03, 0.03, '实线=可靠区 ≥0.5 Hz，虚线=过渡区近似',
                transform=ax.transAxes, fontsize=6.5, alpha=0.7)
    for index in range(n_temp, nrows * ncols):   # 关闭多余子图
        axes[index // ncols][index % ncols].axis('off')
    fig.suptitle('DFN 小信号 EIS 仿真 · 标准 Nyquist 图'
                 '（Z\' vs -Z\'\'，等比例；不同 SOC 与温度）', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f'[绘图] Nyquist 图: {out_png}')


def plot_fit_trends(fit_records, out_png):
    """等效电路参数趋势：R0/Rct 随 SOC 变化（多温度曲线）"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)
    for temperature, group in fit_records.groupby('temperature_c'):
        group = group.sort_values('soc')
        axes[0].plot(group['soc'] * 100, group['r0_ohm'], 'o-',
                     label=f'{temperature:.1f} ℃')
        axes[1].plot(group['soc'] * 100, group['rct_ohm'], 'o-',
                     label=f'{temperature:.1f} ℃')
    axes[0].set_xlabel('SOC [%]')
    axes[0].set_ylabel('R0 [Ω]')
    axes[0].set_title('欧姆内阻 R0 vs SOC')
    axes[1].set_xlabel('SOC [%]')
    axes[1].set_ylabel('Rct [Ω]')
    axes[1].set_title('电荷转移电阻 Rct vs SOC（含 SEI 膜阻贡献）')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f'[绘图] 参数趋势图: {out_png}')


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _refit_existing():
    """EIS_REFIT 模式：读已有 eis_impedance.csv 重拟合参数（不重新仿真）"""
    if not os.path.exists(IMPEDANCE_CSV_PATH):
        raise FileNotFoundError(f'未找到 {IMPEDANCE_CSV_PATH}，请先完整运行一次 EIS 仿真')
    pybamm.set_logging_level('WARNING')
    records = pd.read_csv(IMPEDANCE_CSV_PATH)
    fit_rows = []
    for (temperature_c, soc), group in records.groupby(['temperature_c', 'soc']):
        valid = np.isfinite(group['z_real_ohm']) & np.isfinite(group['z_imag_ohm'])
        if int(valid.sum()) < 6:
            print(f'[跳过] {temperature_c:.1f} ℃ / SOC {soc:.0%}：有效频点不足')
            continue
        frequencies = group['frequency_hz'][valid].to_numpy(dtype=float)
        impedance = (group['z_real_ohm'][valid].to_numpy(dtype=float)
                     + 1.0j * group['z_imag_ohm'][valid].to_numpy(dtype=float))
        fit = fit_zarc(frequencies, impedance)
        if fit is None:
            print(f'[跳过] {temperature_c:.1f} ℃ / SOC {soc:.0%}：拟合失败')
            continue
        fit_rows.append({'temperature_c': float(temperature_c),
                         'soc': float(soc), **fit})
        print(f'[重拟合] {temperature_c:.1f} ℃ / SOC {soc:.0%}: '
              f"R0={fit['r0_ohm']:.3g} Ω, Rct={fit['rct_ohm']:.3g} Ω, "
              f"Q={fit['cpe_q']:.3g}, n={fit['cpe_n']:.3f}, "
              f"σ={fit['sigma_warburg']:.3g}, "
              f"RMSE={fit['rmse_ohm']:.2e} Ω")
    fit_records = pd.DataFrame(fit_rows)
    fit_records.to_csv(FIT_CSV_PATH, index=False)
    print(f'[输出] 重拟合参数表: {FIT_CSV_PATH}（{len(fit_records)} 行）')
    trends_png = os.path.join(OUT_DIR, 'eis_fit_trends.png')
    plot_fit_trends(fit_records, trends_png)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if EIS_REFIT:
        _refit_existing()
        return
    pybamm.set_logging_level('WARNING')
    for env_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(env_key, '1')

    temperatures = list(ALL_TEMPERATURES) if TEMPERATURES_C is None else list(TEMPERATURES_C)
    frequencies = np.asarray(FREQUENCIES_HZ, dtype=float)
    # C-rate 电流 = 容量[Ah] × C-rate（1C = 0.012 A）；C/200 → 0.06 mA
    amplitude_a = AMPLITUDE_C_RATE * NOMINAL_CAPACITY

    mode_tag = '（EIS_QUICK 冒烟）' if EIS_QUICK else ''
    print(f'[配置{mode_tag}] 模型选项 {EIS_MODEL_OPTIONS}')
    print(f'[配置] 温度 {len(temperatures)} 档（{temperatures[0]:.1f} ~ '
          f'{temperatures[-1]:.1f} ℃）× SOC {len(SOC_LEVELS)} 档 × '
          f'频率 {len(frequencies)} 点（{frequencies[0]:.4g} ~ '
          f'{frequencies[-1]:.4g} Hz）')
    print(f'[配置] 激励振幅 C/{1 / AMPLITUDE_C_RATE:.0f} = {amplitude_a * 1000:.3f} mA；'
          f'每频点 max({PERIODS_PER_FREQ}~{LOW_FREQ_PERIODS} 周期, {MIN_WARMUP_S} s)，'
          f'取最后 {ANALYSIS_PERIODS} 个完整周期做基波提取；频点间链式衔接')
    if not os.path.exists(OCV_TABLE_PATH):
        print('[提醒] 未找到 OCV 先验表，SOC→化学计量比使用线性近似'
              '（建议先运行 si_halfcell_dataset.py）')

    rows = []
    fit_rows = []
    wall_start = time.time()
    for temperature_c in temperatures:
        for soc in SOC_LEVELS:
            job_start = time.time()
            impedance = run_eis_job(temperature_c, soc, frequencies, amplitude_a)
            for frequency, z in zip(frequencies, impedance):
                rows.append({
                    'temperature_c': float(temperature_c),
                    'soc': float(soc),
                    'frequency_hz': float(frequency),
                    'z_real_ohm': float(z.real),
                    'z_imag_ohm': float(z.imag),
                    'z_mag_ohm': float(abs(z)),
                    'z_phase_deg': float(np.degrees(np.angle(z))),
                })
            valid = np.isfinite(impedance.real) & np.isfinite(impedance.imag)
            fit = (fit_zarc(frequencies[valid], impedance[valid])
                   if valid.sum() >= 6 else None)
            if fit is not None:
                fit_rows.append({'temperature_c': float(temperature_c),
                                 'soc': float(soc), **fit})
                status = (f"R0={fit['r0_ohm']:.3g} Ω, Rct={fit['rct_ohm']:.3g} Ω, "
                          f"Q={fit['cpe_q']:.3g}, n={fit['cpe_n']:.3f}, "
                          f"σ={fit['sigma_warburg']:.3g}, "
                          f"RMSE={fit['rmse_ohm']:.2e} Ω")
            else:
                status = '等效电路拟合跳过（有效频点不足）'
            print(f'[完成] {temperature_c:.1f} ℃ / SOC {soc:.0%}: '
                  f'{int(valid.sum())}/{len(frequencies)} 频点有效, '
                  f'耗时 {time.time() - job_start:.0f} s, {status}')

    record_path = IMPEDANCE_CSV_PATH
    records = pd.DataFrame(rows)
    records.to_csv(record_path, index=False)

    fit_path = FIT_CSV_PATH
    fit_records = pd.DataFrame(fit_rows)
    if len(fit_records):
        fit_records.to_csv(fit_path, index=False)

    nyquist_png = os.path.join(OUT_DIR, 'eis_nyquist.png')
    plot_nyquist(records, nyquist_png, temperatures)

    trends_png = None
    if len(fit_records):
        trends_png = os.path.join(OUT_DIR, 'eis_fit_trends.png')
        plot_fit_trends(fit_records, trends_png)

    print('\n[输出] EIS 仿真产物:')
    print('  ', record_path, f'({len(records)} 行，温度×SOC×频率长表)')
    if len(fit_records):
        print('  ', fit_path, f'({len(fit_records)} 行，等效电路参数)')
    print('  ', nyquist_png)
    if trends_png:
        print('  ', trends_png)
    print(f'\n[总计] 总耗时 {(time.time() - wall_start) / 60:.1f} min')


if __name__ == '__main__':
    main()
