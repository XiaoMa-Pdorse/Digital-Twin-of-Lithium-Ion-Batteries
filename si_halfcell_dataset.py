# -*- coding: utf-8 -*-
"""Si（非晶硅）半电池训练数据集生成脚本：一份统一数据集同时服务 SOC 与 SOH 训练。

统一数据集 ``data/si-c-half-cell/``
-----------------------------------
- 电化学模型升级为 **DFN（Doyle-Fuller-Newman，真 P2D）**：同时求解固相颗粒
  扩散 PDE、液相（电解液）扩散 PDE 与 Butler-Volmer 动力学，替代原 SPMe 的
  电解液浓度均匀假设（高倍率/厚电极下失效）；
- **电化学-热耦合**：``thermal='lumped'`` 集总热模型（含不可逆欧姆/反应热与
  可逆熵热），电池温度成为求解变量——导出温度列为真实电池温度（随充放电
  动态变化），而非原脚本的恒定环境温度；
- **电化学-力耦合 + 裂纹老化**：``particle mechanics='swelling and cracking'``
  颗粒膨胀-开裂力学 + ``SEI on cracks='true'`` 裂纹处 SEI 持续生长 +
  ``loss of active material='stress-driven'`` 应力驱动活性材料损失——硅负极
  体积膨胀导致的「应力→裂纹→裂纹处 SEI→容量衰减」链条完整落地；
- SEI 老化保留（溶剂扩散限制型 SEI + 分布膜阻 + 孔隙率变化，参数取自
  OKane2022_graphite_SiOx_halfcell 默认值）：充电容量随循环平滑衰减，
  容量表 SOH = 本圈充电容量 / 该电池最大充电容量 为真实老化曲线；
- **温度网格：25.0 ~ 40.0 ℃，步长 0.5 ℃，共 31 只电池**（编号 000 ~ 030）：
  温度通过 Arrhenius 项同时影响 SEI 生长速率与动力学，形成「温度-衰减快慢」
  梯度，供 SOH/RUL 模型学习并做高温外推验证；
- 每只电池 3 圈 C/20 化成（仅用于生成 OCV 先验表，不导出）+ 50 圈 C/2
  （导出），每条半循环曲线按容量分数均匀重采样约 512 点；
- SOC 与 SOH 训练共用同一份 CSV：
  * SOC_Train.py 取 line 40（放电段）：容量分数标签由该圈充电容量归一化，
    老化不破坏标签自洽性；SOH 列作为输入特征携带真实衰减信息；
  * SOH_Train.py 取 line 37（充电段）：容量表 SOH 列为真实老化曲线；
  * SOH 侧 CNN 含 4 层 kernel=32 的 Conv1D 与 4 次 MaxPooling1D，要求
    输入长度 >= 481，512 为满足条件的 2 的幂；
- 只导出 C/2 阶段的 50 圈：前 3 圈化成的容量差异由倍率（快充极化）而非
  老化引起，若混入会使 SOH 出现断层，语义失真；
- 附带产物：
  * ``si_c_half_cell_soc_train.csv``：000 号电池第 3 圈（最后一圈低倍率
    准 OCV 循环）放电曲线的 50 点重采样 (SOC [%], Voltage [V],
    Stoichiometry x)，作为 kalman_soc.py 的 OCV-SOC 先验表；
  * ``coupled_diagnostics.csv``：逐圈逐段（放电/充电）耦合物理场诊断表——
    最高温度、平均热源、颗粒表面切向应力、裂纹长度/扩展速率、活性材料
    残留分数、SEI 厚度——用于机制分析、报告与 Streamlit 展示；
  * ``capacity_fade_cell_000.png``：000 号电池循环-容量曲线 PNG，
    可视化 SEI+裂纹老化带来的容量衰减。

采样与补零说明（SOC 时序滑窗）
------------------------------
- 常规圈点数在 511/512 间随圈号奇偶微变（避免所有圈严格等长时
  np.array(list, dtype=object) 退化为规整三维数组、破坏处理器逐圈处理逻辑）；
- 030 号电池第 50 圈额外加 16 点（528 点，全数据集唯一最长圈），配合
  处理器按全局最长循环补零，使除该圈外每个循环之后都补有 >= 16 行零值，
  create_sequence_data 的时序滑窗（TIME_STEPS=8）不会跨循环、跨电池；
  因此编号最大的电池（030）需保持在训练/测试名单的最后一位
  （SOC_Train.py / SOH_Train.py 的 _si_test_indices 不含 030）；
- 该末圈额外点数对 SOH 训练无影响：卷积输入仅多出 <= 16 行补零。

共用要素
--------
- 31 只电池与温度梯度（25.0 ~ 40.0 ℃，步长 0.5 ℃）；
- 几何与电化学参数见 _base_parameter_values（25 μm 电极、6 μm 颗粒、
  硅 OCP/交换电流密度等）；
- 多进程并行仿真（环境变量 SI_HALFCELL_WORKERS 可覆盖进程数，
  SOH_HALFCELL_WORKERS 为兼容旧脚本的备用名），完整生成一次耗时较长
  （DFN 全耦合约为原 SPMe 的 5~20 倍，建议先用 SI_HALFCELL_DEBUG=1 冒烟验证）。

模型级别与冒烟模式（环境变量）
------------------------------
- ``SI_HALFCELL_MODEL``：full（默认，DFN+热+力+裂纹）/ no-mechanics（去力学）
  / no-thermal（去热耦合）/ baseline（仅 SEI，复现原 SPMe 选项），排障降级用；
- ``SI_HALFCELL_DEBUG=1``：仅仿真前 2 只电池、1 圈化成 + 4 圈 C/2，输出到
  ``data/_debug_si_halfcell/``，并打印参数自检与耦合变量清单，用于快速验证
  收敛性，不污染正式数据集。

其他说明
--------
- 硅 OCP 采用 PyBaMM 内置 Chen2020_composite 中的 Mark2016 拟合
  （硅嵌锂/脱锂多项式拟合，出处：Verbrugge et al., J. Electrochem. Soc.
  163(2) A262, 2015）。若需替换为自备 OCP 数据表（例如 Chevrier & Dahn
  2009 的数字化数据），将 ``CUSTOM_OCP_CSV`` 指向两列 (x_LiSi, V) 的 CSV 即可。
- 化学计量比按 Li_xSi 表示：x=0.02 接近空硅，x=3.75 对应 Li15Si4 满嵌锂。
- 几何沿用交接说明（25 μm 电极厚度、6 μm 颗粒等）；最大锂浓度使用硅体系
  278000 mol/m3（Chen2020_composite），而非石墨的 298000 mol/m3。
- EIS 阻抗仿真由独立脚本 ``eis_simulation.py`` 提供（复用本文件的参数集与
  模型选项，小信号正弦扫频 → Nyquist → 等效电路拟合）。
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pybamm

# ---------------------------------------------------------------------------
# 配置区
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'si-c-half-cell')
DEBUG_OUT_DIR = os.path.join(REPO_ROOT, 'data', '_debug_si_halfcell')

# --- 循环协议（半电池文档）：低倍率 0.05C 循环 3 圈 → 正常倍率 0.5C 循环 50 圈 ---
LOW_RATE_C_RATE = 'C/20'   # 低倍率 0.05C（化成 / 准 OCV 循环）
LOW_RATE_CYCLES = 3        # 低倍率循环圈数
HIGH_RATE_C_RATE = 'C/2'   # 正常倍率 0.5C
HIGH_RATE_CYCLES = 50      # 正常倍率循环圈数

# --- 电池温度网格：25.0 ~ 40.0 ℃，步长 0.5 ℃，共 31 只电池（编号 000 ~ 030）---
TEMP_START_C = 25.0
TEMP_STOP_C = 40.0
TEMP_STEP_C = 0.5

V_LOW = 0.06               # 放电（嵌锂）截止电压，非晶硅 OCP 最低约 0.05 V
V_HIGH = 1.5               # 充电（脱锂）截止电压
UNIFIED_POINTS = 512       # 每条半循环重采样点数（CNN 4xConv(k=32)+Pool 要求 >= 481）
OCV_TABLE_POINTS = 50      # OCV 先验表（si_c_half_cell_soc_train.csv）重采样点数
FINAL_CYCLE_EXTRA = 16     # 030 号电池末圈额外点数（保证 SOC 滑窗补零 >= 16，见文首）

STO_INIT = 0.005           # 初始锂化度（x ≈ 0.019，对应放电起点 SOC ≈ 100%）
X_FULL = 3.75              # Li15Si4 对应的 Li_xSi 计量比上限
C_MAX = 278000.0           # 硅最大锂浓度 [mol.m-3]（Chen2020_composite）
NOMINAL_CAPACITY = 0.012   # 标称容量 [A.h]（≈本几何实际电极容量 10.5 mAh，保证倍率时序合理）


# ---------------------------------------------------------------------------
# 模型选项：DFN（真 P2D）+ 电化学-热-力耦合 + 裂纹老化
# ---------------------------------------------------------------------------
# 全耦合（默认）：SEI（溶剂扩散限制型 + 分布膜阻 + 孔隙率变化）+ 颗粒膨胀开裂
# 力学 + 裂纹处 SEI 生长 + 应力驱动 LAM + 集总热模型。
# 说明：
# - 热模型用 'thermal': 'lumped'（集总热）：外部散热由参数集自带的 Total heat
#   transfer coefficient / Cell cooling surface area / Cell volume 控制；
#   不用 'x-lumped'，避免在二维电流收集器下缺失集流器传热参数；
# - lumped 热模型把锂金属对电极与负集流器也纳入热容计算，halfcell 参数集
#   未提供对应物性，需在 _base_parameter_values 中补 5 个参数（见该函数）；
# - 裂纹模型（swelling and cracking）要求颗粒径向网格较密（见 VAR_PTS_OVERRIDE），
#   官方算例 r_p >= 26 才数值稳定；
# - lithium plating 为石墨负极主导机制，与半电池（锂金属对电极）语义不符，不启用。
def _model_options():
    """按 SI_HALFCELL_MODEL 环境变量组装模型选项（full/no-mechanics/no-thermal/baseline）"""
    base = {
        'working electrode': 'positive',
        'SEI': 'solvent-diffusion limited',
        'SEI film resistance': 'distributed',
        'SEI porosity change': 'true',
    }
    mechanics = {
        'particle mechanics': 'swelling and cracking',
        'SEI on cracks': 'true',
        'loss of active material': 'stress-driven',
    }
    thermal = {
        'thermal': 'lumped',
    }
    level = os.environ.get('SI_HALFCELL_MODEL', 'full').strip().lower()
    if level == 'full':
        return {**base, **mechanics, **thermal}
    if level == 'no-mechanics':
        return {**base, **thermal}
    if level == 'no-thermal':
        return {**base, **mechanics}
    if level == 'baseline':
        return dict(base)
    raise ValueError(
        f'未知 SI_HALFCELL_MODEL={level!r}（可选 full/no-mechanics/no-thermal/baseline）')


# 颗粒径向网格覆盖：裂纹模型数值稳定性要求（官方算例 r_p >= 26）。
# 注意：PyBaMM 26.x 的 Simulation var_pts 为整体覆盖而非合并，必须与
# model.default_var_pts 合并使用（否则缺失 x_n 等键会报
# "Points not given for variable 'x_n'"）。
VAR_PTS_OVERRIDE = {'r_p': 30}

# ---------------------------------------------------------------------------
# 耦合物理场诊断：逐圈逐段提取标量表（列名 / 变量候选 / 聚合方式 / 缩放）
# ---------------------------------------------------------------------------
# 变量名以 PyBaMM 官方 DFN/SPM 变量清单为准（如 “X-averaged positive particle
# cracking rate [m.s-1]”），若当前 PyBaMM 版本重命名，debug 模式会打印实际可用
# 变量清单供核对；提取失败的列写 NaN、不中断主流程。
DIAGNOSTIC_SPECS = (
    ('avg_heat_source_w_m3',
     ['X-averaged total heating [W.m-3]', 'Volume-averaged total heating [W.m-3]',
      'Heat source [W.m-3]'], 'mean', 1.0),
    ('max_abs_surface_stress_mpa',
     ['X-averaged positive particle surface tangential stress [Pa]'], 'max_abs', 1e-6),
    ('end_crack_length_um',
     ['X-averaged positive particle crack length [m]'], 'last', 1e6),
    ('avg_cracking_rate_m_s',
     ['X-averaged positive particle cracking rate [m.s-1]'], 'mean_abs', 1.0),
    ('end_active_material_fraction',
     ['X-averaged positive electrode active material volume fraction'], 'last', 1.0),
    ('end_sei_thickness_nm',
     ['X-averaged positive SEI thickness [m]', 'X-averaged SEI thickness [m]'], 'last', 1e9),
)

DIAGNOSTIC_COLUMNS = [
    'test_name', 'cycle', 'phase', 'max_temperature_c', 'end_temperature_c',
    'avg_heat_source_w_m3', 'max_abs_surface_stress_mpa', 'end_crack_length_um',
    'avg_cracking_rate_m_s', 'end_active_material_fraction', 'end_sei_thickness_nm',
]

# 历史保留：baseline 级别的选项组合（与 _model_options()['baseline'] 一致）
SEI_OPTIONS = {
    'working electrode': 'positive',
    'SEI': 'solvent-diffusion limited',
    'SEI film resistance': 'distributed',
    'SEI porosity change': 'true',
}


def build_cells():
    """生成 31 只电池参数：test_name 形如 000-SI-3.0-2500-S（末段 = 温度 × 100）

    温度网格 25.0 ~ 40.0 ℃、步长 0.5 ℃，共 31 档（编号 000 ~ 030）。
    """
    count = int(round((TEMP_STOP_C - TEMP_START_C) / TEMP_STEP_C)) + 1
    cells = []
    for index in range(count):
        temperature_c = TEMP_START_C + TEMP_STEP_C * index
        cells.append({
            'test_name': f'{index:03d}-SI-3.0-{round(temperature_c * 100):04d}-S',
            'temperature': 273.15 + temperature_c,
            'temperature_c': temperature_c,
        })
    return cells


CELLS = build_cells()
CELL_COUNT = len(CELLS)
TRAIN_CELL_INDEX = 0       # OCV 先验表与容量衰减曲线图取自该电池（000 号，25.0 ℃）

# 可选：自备 OCP 数据表路径（两列：x_LiSi, V）。为 None 时用内置 Mark2016 拟合。
CUSTOM_OCP_CSV = None


# ---------------------------------------------------------------------------
# 非晶硅 OCP（Mark2016 拟合，抄自 PyBaMM Chen2020_composite，引用见文首）
# ---------------------------------------------------------------------------
def silicon_ocp_lithiation(sto):
    """硅嵌锂 OCP 拟合（对应 sto=Li/Si 归一化锂化度，0 < sto < 1）"""
    p1, p2, p3, p4 = -96.63, 372.6, -587.6, 489.9
    p5, p6, p7, p8 = -232.8, 62.99, -9.286, 0.8633
    return (
        p1 * sto ** 7 + p2 * sto ** 6 + p3 * sto ** 5 + p4 * sto ** 4
        + p5 * sto ** 3 + p6 * sto ** 2 + p7 * sto + p8
    ) + 1e-4 * (1 / sto + 1 / (sto - 1))


def silicon_ocp_delithiation(sto):
    """硅脱锂 OCP 拟合"""
    p1, p2, p3, p4 = -51.02, 161.3, -205.7, 140.2
    p5, p6, p7, p8 = -58.76, 16.87, -3.792, 0.9937
    return (
        p1 * sto ** 7 + p2 * sto ** 6 + p3 * sto ** 5 + p4 * sto ** 4
        + p5 * sto ** 3 + p6 * sto ** 2 + p7 * sto + p8
    )


def silicon_ocp_average(sto):
    """嵌锂/脱锂平均 OCP（PyBaMM composite 参数集的默认映射）"""
    return (silicon_ocp_lithiation(sto) + silicon_ocp_delithiation(sto)) / 2


def silicon_exchange_current_density(c_e, c_s_surf, c_s_max, T):
    """硅电极交换电流密度（Chen2020_composite 同款）"""
    m_ref = 6.48e-7 * 28700 / 278000  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    return m_ref * arrhenius * c_e ** 0.5 * c_s_surf ** 0.5 * (c_s_max - c_s_surf) ** 0.5


def make_custom_ocp(path):
    """把两列 (x_LiSi, V) 的 CSV 转为 pybamm 可用的 OCP 函数"""
    data = pd.read_csv(path)
    x_sto = data.iloc[:, 0].to_numpy(dtype=float) / X_FULL
    u = data.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(x_sto)

    def ocp(sto):
        return pybamm.Interpolant(x_sto[order], u[order], sto, name='custom Si OCP')

    return ocp


SI_OCP = make_custom_ocp(CUSTOM_OCP_CSV) if CUSTOM_OCP_CSV else silicon_ocp_average


# ---------------------------------------------------------------------------
# 模型与参数（两套数据集共用几何与电化学参数，仅 SEI 子模型不同）
# ---------------------------------------------------------------------------
def _base_parameter_values(temperature):
    """构建两套数据集共用的参数集（几何、OCP、交换电流密度、温度、容量）"""
    parameter_values = pybamm.ParameterValues('OKane2022_graphite_SiOx_halfcell')
    parameter_values.update({
        # --- 电极面积：沿用 test.py（0.01 m × 0.0113 m = 1.13 cm²）---
        'Electrode width [m]': 0.01,
        'Electrode height [m]': 0.0113,
        # --- 正极（工作电极）= 非晶硅，几何沿用交接说明 ---
        'Positive electrode thickness [m]': 25e-6,
        'Positive electrode porosity': 0.4,
        'Positive electrode active material volume fraction': 0.5,
        'Positive particle radius [m]': 6e-6,
        'Maximum concentration in positive electrode [mol.m-3]': C_MAX,
        'Initial concentration in positive electrode [mol.m-3]': C_MAX * STO_INIT,
        'Positive particle diffusivity [m2.s-1]': 1.67e-14,
        'Positive electrode OCP [V]': SI_OCP,
        'Positive electrode exchange-current density [A.m-2]':
            silicon_exchange_current_density,
        # --- 温度与容量 ---
        'Ambient temperature [K]': temperature,
        'Initial temperature [K]': temperature,
        'Nominal cell capacity [A.h]': NOMINAL_CAPACITY,
        # --- 半电池热耦合补丁参数（lumped 热模型把锂金属对电极与负集流器一并纳入
        # 热容/产热分布，但 OKane2022_graphite_SiOx_halfcell 参数集未提供这些物性；
        # 集流器 3 项取自 OKane2022 全电池参数集同源值，锂金属 2 项为文献物性，
        # 仅用于热的闭合计算，不影响电化学参数）---
        'Negative current collector conductivity [S.m-1]': 58411000.0,
        'Negative current collector density [kg.m-3]': 8960.0,
        'Negative current collector specific heat capacity [J.kg-1.K-1]': 385.0,
        'Negative electrode density [kg.m-3]': 534.0,          # 锂金属
        'Negative electrode specific heat capacity [J.kg-1.K-1]': 3570.0,  # 锂金属
        # --- 电压保护边界（步骤终止由 experiment 条件控制） ---
        'Lower voltage cut-off [V]': 0.005,
        'Upper voltage cut-off [V]': V_HIGH,
    })
    return parameter_values


def _experiment_steps(low_cycles, high_cycles):
    """循环协议：先 low_cycles 圈 C/20 化成，再 high_cycles 圈 C/2；每圈含放电、充电两步"""
    return (
        [(f'Discharge at {LOW_RATE_C_RATE} until {V_LOW} V',
          f'Charge at {LOW_RATE_C_RATE} until {V_HIGH} V')] * low_cycles
        + [(f'Discharge at {HIGH_RATE_C_RATE} until {V_LOW} V',
            f'Charge at {HIGH_RATE_C_RATE} until {V_HIGH} V')] * high_cycles
    )


def _make_solver():
    """IDAKLU（快，新版默认）优先，失败退回 Casadi safe 模式（旧版兼容）"""
    try:
        return pybamm.IDAKLUSolver()
    except Exception:  # noqa: BLE001 - 旧版 PyBaMM 无 IDAKLU 时退回 Casadi
        return pybamm.CasadiSolver(mode='safe')


def build_simulation(temperature, options, low_cycles, high_cycles):
    """构建 DFN（P2D）+ 电化学-热-力耦合 + 裂纹老化的半电池仿真

    协议参数由调用方传入（Windows spawn 多进程下不依赖主进程全局变量修改）。
    var_pts 必须与模型默认网格合并（PyBaMM 26.x 为整体覆盖语义）。
    """
    model = pybamm.lithium_ion.DFN(options)
    return pybamm.Simulation(
        model,
        parameter_values=_base_parameter_values(temperature),
        experiment=pybamm.Experiment(_experiment_steps(low_cycles, high_cycles)),
        var_pts={**model.default_var_pts, **VAR_PTS_OVERRIDE},
        solver=_make_solver(),
    )


# ---------------------------------------------------------------------------
# 逐圈数据提取与重采样
# ---------------------------------------------------------------------------
def _get_series(cycle_solution, candidates):
    """按候选名依次尝试，返回 (数值数组, 实际变量名)"""
    last_error = None
    for name in candidates:
        try:
            return np.asarray(cycle_solution[name].entries), name
        except Exception as error:  # noqa: BLE001 - 变量不存在时换下一个候选
            last_error = error
    raise KeyError(f'未找到变量候选 {candidates}: {last_error}')


def _extract_optional_series(step_solution, candidates):
    """按候选名返回第一个可用变量的数值数组，全部失败返回 (None, None)"""
    for name in candidates:
        try:
            return np.asarray(step_solution[name].entries, dtype=float), name
        except Exception:  # noqa: BLE001 - 变量缺失/形状异常时尝试下一个候选
            continue
    return None, None


def extract_runs(cycle_solution, ambient_temperature_c=None):
    """把一整个循环拆成放电/充电两段，返回各段的时间、电压、电流、容量、计量比

    半电池模型的电流符号与全电池相反（Discharge 步骤的电流为正），不能按电流
    正负切分，因此按 experiment 步骤顺序取：steps[0]=Discharge、steps[1]=Charge。
    ambient_temperature_c 为温度变量缺失时的兜底值（热耦合关闭时用）。
    """
    steps = cycle_solution.steps
    if len(steps) < 2:
        raise RuntimeError(f'循环步骤数不足，无法切分放电/充电段: {len(steps)}')
    discharge, sto_name = _pack_segment(steps[0], ambient_temperature_c)
    charge, _ = _pack_segment(steps[1], ambient_temperature_c)
    return discharge, charge, sto_name


def _pack_segment(step_solution, ambient_temperature_c=None):
    """把单个 experiment 步骤整理为按时间排布的各通道数组

    新增温度通道：热耦合开启时取仿真电池温度（X-averaged cell temperature），
    作为数据集 temperature 列（真实动态温度）；变量缺失时退化到环境温度。
    """
    t = _get_series(step_solution, ['Time [s]'])[0]
    voltage = _get_series(step_solution, ['Voltage [V]'])[0]
    current = _get_series(step_solution, ['Current [A]'])[0]

    sto_values, sto_name = _get_series(step_solution, [
        'X-averaged positive particle surface stoichiometry',
        'X-averaged positive particle surface concentration [mol.m-3]',
        'X-averaged positive particle stoichiometry',
    ])
    # 若取到的是颗粒径向分布 (r, t)，取表面点
    if sto_values.ndim == 2:
        sto_values = (sto_values[-1, :] if sto_values.shape[-1] == len(t)
                      else sto_values[:, -1])
    if 'stoichiometry' not in sto_name:
        sto_values = sto_values / C_MAX

    temperature_c, _ = _extract_optional_series(step_solution, [
        'X-averaged cell temperature [K]',
        'Volume-averaged cell temperature [K]',
        'Cell temperature [K]',
    ])
    if temperature_c is None:
        fallback = (float(ambient_temperature_c)
                    if ambient_temperature_c is not None else np.nan)
        temperature_c = np.full(len(t), fallback)
    else:
        temperature_c = np.atleast_1d(temperature_c) - 273.15
        if temperature_c.ndim == 2:   # (x, t) 分布时沿空间取平均
            temperature_c = (temperature_c.mean(axis=0)
                             if temperature_c.shape[-1] == len(t)
                             else temperature_c.mean(axis=1))
        if temperature_c.size == 1:  # 集总标量温度广播到各时刻
            temperature_c = np.full(len(t), float(temperature_c.reshape(-1)[0]))

    t_local = t - t[0]
    q_local = np.concatenate(
        [[0.0], np.cumsum(-current[1:] * np.diff(t_local))]) / 3600.0  # [A.h]
    wh_local = np.concatenate(
        [[0.0], np.cumsum(voltage[1:] * (-current[1:])
                          * np.diff(t_local))]) / 3600.0  # [W.h]
    return {
        't_h': t_local / 3600.0,
        'voltage': voltage,
        'current': current,
        'capacity': np.abs(q_local),
        'wh': np.abs(wh_local),
        'sto': sto_values,
        'temp_c': temperature_c,
        # 保留 Solution 引用供耦合诊断提取（simulate_cell 用完即 pop 释放）
        'solution': step_solution,
    }, sto_name


def resample_run(run, n_points):
    """把一段曲线按容量分数均匀重采样为 n_points 个点（含温度通道）"""
    q = run['capacity']
    q_end = q[-1]
    if q_end <= 0:
        raise RuntimeError('容量非正，无法重采样')
    frac = np.linspace(0.0, 1.0, n_points)
    q_grid = frac * q_end
    return {
        'frac': frac,
        't_h': np.interp(q_grid, q, run['t_h']),
        'voltage': np.interp(q_grid, q, run['voltage']),
        'current': np.interp(q_grid, q, run['current']),
        'capacity': q_grid,
        'wh': np.interp(q_grid, q, run['wh']),
        'sto': np.interp(q_grid, q, run['sto']),
        'temp_c': np.interp(q_grid, q, run['temp_c']),
    }


def unified_point_count(cell_index, cycle_no, cell_count, high_cycles,
                        base_points=UNIFIED_POINTS,
                        final_extra=FINAL_CYCLE_EXTRA):
    """统一数据集每个循环（C/2 段，编号 1~high_cycles）的重采样点数

    常规圈在 511/512 间随圈号奇偶微变（防 np.array(object) 退化，见文首）。
    编号最大电池（cell_index == cell_count - 1，即 030 号）的最后一圈额外加
    final_extra 点（528，全数据集唯一最长圈）：处理器按全局最长循环补零后，
    除该圈外每个循环之后都补有 >= FINAL_CYCLE_EXTRA 行零值，SOC_Train.py 的
    create_sequence_data 时序滑窗（TIME_STEPS=8）不会跨循环、跨电池（因此
    编号最大的电池需保持在训练/测试名单的最后一位）。debug 模式 final_extra=0。
    """
    if final_extra > 0 and cycle_no == high_cycles and cell_index == cell_count - 1:
        return base_points + final_extra
    return base_points - (cycle_no % 2)


# ---------------------------------------------------------------------------
# 数据导出
# ---------------------------------------------------------------------------
CYCLE_COLUMNS = [
    'test_name', 'record_id', 'time', 'step_time', 'line', 'voltage', 'current',
    'charging_capacity', 'discharging_capacity', 'wh_charging', 'wh_discharging',
    'temperature', 'cycle_count',
]
CAPACITY_COLUMNS = CYCLE_COLUMNS + ['max_temperature', 'average_tension']


def export_soc_train_csv(cell_name, discharge_run, out_dir=None):
    """导出 50 行 (SOC [%], Voltage [V], Stoichiometry x)，SOC 从 0% 递增到 100%

    discharge_run 传入原始（未重采样）放电段曲线：000 号电池第 3 圈
    （最后一圈低倍率准 OCV 循环）的放电数据，作为 kalman_soc.py 的
    OCV-SOC 先验表来源。"""
    discharge_grid = resample_run(discharge_run, OCV_TABLE_POINTS)
    soc_percent = (1.0 - discharge_grid['frac']) * 100.0
    table = pd.DataFrame({
        'SOC [%]': soc_percent[::-1],
        'Voltage [V]': discharge_grid['voltage'][::-1],
        'Stoichiometry x': discharge_grid['sto'][::-1] * X_FULL,
    })
    path = os.path.join(out_dir or OUT_DIR, 'si_c_half_cell_soc_train.csv')
    table.to_csv(path, index=False)
    return path


def build_unibo_tables(cells_runs, out_dir=None):
    """把各电池 C/2 阶段的逐圈曲线写成 UNIBO 同构的 cycle / capacity 表（统一数据集）

    cell['grids'] 为该电池逐圈 (放电重采样网格, 充电重采样网格) 的列表。
    含 SEI + 裂纹老化：charging_capacity 列写每圈真实充电容量（随循环衰减），
    既是容量表 SOH（本圈充电容量 / 该电池最大充电容量）的真实老化依据，
    也是 SOC_Train.py 容量分数标签的归一化基准。
    温度列写仿真电池真实温度（热耦合下随充放电动态变化）；capacity 表的
    temperature 为段末温度、max_temperature 为段内最高温。
    """
    cycle_frames = []
    capacity_rows = []
    record_id = 0

    for cell in cells_runs:
        test_name = cell['test_name']
        time_cum_h = 0.0
        cell_rows = []
        for cycle_no, (discharge, charge) in enumerate(cell['grids'], start=1):
            cycle_charge_capacity = charge['capacity'][-1]
            cycle_charge_wh = charge['wh'][-1]
            for line, grid in ((40, discharge), (37, charge)):
                is_discharge = line == 40
                # UNIBO 约定：放电电流为负、充电电流为正（半电池仿真中放电步骤电流为正）
                current_sign = -1.0 if is_discharge else 1.0
                n_points = len(grid['frac'])
                for point in range(n_points):
                    record_id += 1
                    cell_rows.append({
                        'test_name': test_name,
                        'record_id': record_id,
                        'time': time_cum_h + grid['t_h'][point],
                        'step_time': grid['t_h'][point],
                        'line': line,
                        'voltage': grid['voltage'][point],
                        'current': current_sign * abs(grid['current'][point]),
                        'charging_capacity': (cycle_charge_capacity if is_discharge
                                              else grid['capacity'][point]),
                        'discharging_capacity': (grid['capacity'][point]
                                                 if is_discharge else 0.0),
                        'wh_charging': (cycle_charge_wh if is_discharge
                                        else grid['wh'][point]),
                        'wh_discharging': (grid['wh'][point]
                                           if is_discharge else 0.0),
                        'temperature': grid['temp_c'][point],
                        'cycle_count': cycle_no,
                    })
                # 与 UNIBO 一致：capacity 表保存该 run 的最后一条记录
                capacity_rows.append({
                    'test_name': test_name,
                    'record_id': record_id,
                    'time': time_cum_h + grid['t_h'][-1],
                    'step_time': grid['t_h'][-1],
                    'line': line,
                    'voltage': grid['voltage'][-1],
                    'current': current_sign * abs(grid['current'][-1]),
                    'charging_capacity': cycle_charge_capacity,  # 本圈真实充电容量
                    'discharging_capacity': (grid['capacity'][-1]
                                             if is_discharge else 0.0),
                    'wh_charging': cycle_charge_wh,
                    'wh_discharging': (grid['wh'][-1]
                                       if is_discharge else 0.0),
                    'temperature': float(grid['temp_c'][-1]),
                    'cycle_count': cycle_no,
                    'max_temperature': float(np.max(grid['temp_c'])),
                    'average_tension': float(np.mean(grid['voltage'])),
                })
                time_cum_h += grid['t_h'][-1]
        cycle_frames.append(pd.DataFrame(cell_rows, columns=CYCLE_COLUMNS))

    cycle_path = os.path.join(out_dir or OUT_DIR, 'test_result.csv')
    capacity_path = os.path.join(out_dir or OUT_DIR, 'test_result_trial_end.csv')
    cycle_table = pd.concat(cycle_frames, ignore_index=True)
    cycle_table.to_csv(cycle_path, index=False)
    pd.DataFrame(capacity_rows, columns=CAPACITY_COLUMNS).to_csv(capacity_path, index=False)
    return cycle_path, capacity_path, len(cycle_table), len(capacity_rows)


def _reduce_series(values, mode, scale):
    """按聚合方式把一个时间序列压成一个标量（None/空值 -> NaN）"""
    if values is None:
        return np.nan
    series = np.asarray(values, dtype=float)
    if series.ndim == 2:          # 仍有空间维时先沿空间维平均
        series = series.mean(axis=0)
    series = series.ravel()
    if series.size == 0:
        return np.nan
    if mode == 'mean':
        result = np.nanmean(series)
    elif mode == 'mean_abs':
        result = np.nanmean(np.abs(series))
    elif mode == 'last':
        result = series[-1]
    elif mode == 'max_abs':
        result = np.nanmax(np.abs(series))
    else:
        raise ValueError(f'未知聚合方式 {mode!r}')
    return float(result) * scale


def make_diagnostic_row(run, test_name, cycle_no, phase):
    """把一段（放电/充电）的耦合物理场标量整理成诊断表一行

    run 为 _pack_segment 的输出（含 'solution' 引用与 'temp_c' 温度通道）；
    温度类两列直接由温度通道得出，其余列按 DIAGNOSTIC_SPECS 提取，
    变量缺失时写 NaN（不中断主流程）。
    """
    row = {
        'test_name': test_name,
        'cycle': cycle_no,
        'phase': phase,
        'max_temperature_c': float(np.max(run['temp_c'])),
        'end_temperature_c': float(run['temp_c'][-1]),
    }
    step_solution = run.get('solution')
    for column, candidates, mode, scale in DIAGNOSTIC_SPECS:
        values, _name = _extract_optional_series(step_solution, candidates)
        row[column] = _reduce_series(values, mode, scale)
    return row


def _print_debug_variables(step_solution):
    """debug 模式：打印与耦合物理场相关的实际变量清单，便于核对变量名"""
    try:
        names = list(step_solution.all_models[0].variables.keys())
    except Exception:  # noqa: BLE001 - 拿不到变量表时静默跳过
        return
    keywords = ('crack', 'stress', 'active material', 'temperature',
                'heating', 'sei thickness')
    hits = sorted({name for name in names
                   if any(keyword in name.lower() for keyword in keywords)})
    print('[debug] 耦合物理场相关可用变量:')
    for name in hits:
        print('   -', name)


def write_coupled_diagnostics(cells_runs, out_dir):
    """把逐圈逐段（放电/充电）的耦合物理场诊断标量写成 coupled_diagnostics.csv"""
    rows = []
    for cell in cells_runs:
        rows.extend(cell['diagnostics'])
    path = os.path.join(out_dir, 'coupled_diagnostics.csv')
    pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS).to_csv(path, index=False)
    return path, len(rows)


def plot_capacity_fade(cell_runs, output_path):
    """输出一颗电池的循环-容量曲线（PNG），可视化 SEI 老化带来的容量衰减

    x 轴为 C/2 阶段循环圈数（1 ~ 50），y 轴为该圈放电/充电容量（mAh）。
    """
    import matplotlib
    try:
        matplotlib.use('Agg', force=True)  # 只写文件，不弹窗阻塞脚本退出
    except Exception:  # noqa: BLE001 - 后端已固定时忽略
        pass
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    discharge_caps = [discharge['capacity'][-1] * 1000.0
                      for discharge, _ in cell_runs['grids']]
    charge_caps = [charge['capacity'][-1] * 1000.0
                   for _, charge in cell_runs['grids']]
    cycles = np.arange(1, len(charge_caps) + 1)
    fade = (1.0 - charge_caps[-1] / charge_caps[0]) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.plot(cycles, discharge_caps, 'o-', markersize=3.5, linewidth=1.4,
            color='#1f77b4', label='放电容量')
    ax.plot(cycles, charge_caps, 's--', markersize=3.5, linewidth=1.4,
            color='#d62728', label='充电容量')
    ax.set_xlabel('循环圈数（C/2 阶段）')
    ax.set_ylabel('容量 [mAh]')
    ax.set_title(f"Si 半电池 {cell_runs['test_name']} 循环-容量曲线"
                 f"（SEI+裂纹老化，{len(charge_caps)} 圈衰减 {fade:.1f}%）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f'[绘图] 循环-容量曲线: {output_path}')
    print(f'[绘图] 首圈充电容量 {charge_caps[0]:.3f} mAh -> 末圈 '
          f'{charge_caps[-1]:.3f} mAh，衰减 {fade:.1f}%')
    return fade


# ---------------------------------------------------------------------------
# 多进程 worker
# ---------------------------------------------------------------------------
def simulate_cell(cell_params):
    """多进程 worker：仿真一只电池（DFN+热-力耦合+裂纹老化），返回逐圈曲线与诊断

    全部圈数仿真（化成段 + C/2 段）；化成段不导出，仅保留 000 号电池最后
    一圈化成的原始放电曲线用于生成 OCV 先验表。协议参数（圈数、点数、模型
    选项）全部由 cell_params 传入：Windows spawn 模式下子进程无主进程全局
    变量修改，必须显式传参。
    """
    pybamm.set_logging_level('ERROR')
    start = time.time()
    low_cycles = cell_params['low_rate_cycles']
    high_cycles = cell_params['high_rate_cycles']
    ambient_c = cell_params['temperature_c']

    simulation = build_simulation(cell_params['temperature'],
                                  cell_params['model_options'],
                                  low_cycles, high_cycles)
    simulation.solve()
    all_runs = [extract_runs(cycle_solution, ambient_c)
                for cycle_solution in simulation.solution.cycles]

    # 释放化成段 Solution 引用（仅保留 000 号电池末圈准 OCV 放电段，见文首）
    for index, (discharge, charge, _) in enumerate(all_runs[:low_cycles]):
        keep_discharge = (cell_params['cell_index'] == TRAIN_CELL_INDEX
                          and index == low_cycles - 1)
        if not keep_discharge:
            discharge.pop('solution', None)
        charge.pop('solution', None)

    high_rate_runs = all_runs[low_cycles:]
    if len(high_rate_runs) != high_cycles:
        raise RuntimeError(
            f'C/2 阶段循环数 {len(high_rate_runs)} != 期望 {high_cycles}')

    grids = []
    diagnostics = []
    debug_printed = False
    for cycle_no, (discharge, charge, _) in enumerate(high_rate_runs, start=1):
        n_points = unified_point_count(
            cell_params['cell_index'], cycle_no, cell_params['cell_count'],
            high_cycles, cell_params['unified_points'],
            cell_params['final_cycle_extra'])
        grids.append((resample_run(discharge, n_points),
                      resample_run(charge, n_points)))
        # 耦合物理场诊断（逐圈逐段），提取后立即释放 Solution 引用控制内存
        for phase, run in (('discharge', discharge), ('charge', charge)):
            if cell_params.get('debug') and not debug_printed:
                _print_debug_variables(run.get('solution'))
                debug_printed = True
            diagnostics.append(make_diagnostic_row(
                run, cell_params['test_name'], cycle_no, phase))
        discharge.pop('solution', None)
        charge.pop('solution', None)

    charge_caps = [charge['capacity'][-1] * 1000.0 for _, charge in grids]
    diagnostics_missing = [column for column, *_ in DIAGNOSTIC_SPECS
                           if all(np.isnan(row[column]) for row in diagnostics)]

    quasi_ocv = None
    if cell_params['cell_index'] == TRAIN_CELL_INDEX:
        quasi_ocv = all_runs[low_cycles - 1][0]
        quasi_ocv.pop('solution', None)
    return {
        'test_name': cell_params['test_name'],
        'temperature': cell_params['temperature'],
        'temperature_c': ambient_c,
        'grids': grids,
        'diagnostics': diagnostics,
        'diagnostics_missing': diagnostics_missing,
        'elapsed_s': time.time() - start,
        'first_charge_cap_mah': charge_caps[0],
        'last_charge_cap_mah': charge_caps[-1],
        'soh_series': [q / max(charge_caps) for q in charge_caps],
        # OCV 先验表所需：000 号电池末圈低倍率循环的原始放电曲线
        'quasi_ocv_discharge': quasi_ocv,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _worker_count():
    """并行进程数：SI_HALFCELL_WORKERS 优先，SOH_HALFCELL_WORKERS 兼容旧脚本"""
    raw = (os.environ.get('SI_HALFCELL_WORKERS', '')
           or os.environ.get('SOH_HALFCELL_WORKERS', ''))
    try:
        count = int(raw or 0)
    except ValueError:
        count = 0
    return count or max(1, min(12, (os.cpu_count() or 4) - 1))


def _build_profiles():
    """返回运行配置：(cells, low_cycles, high_cycles, base_points, final_extra, out_dir, debug)

    SI_HALFCELL_DEBUG=1 时使用精简协议（2 只电池、1 圈化成 + 4 圈 C/2）并输出
    到独立目录 data/_debug_si_halfcell/，不污染正式数据集。
    """
    debug = os.environ.get('SI_HALFCELL_DEBUG', '').strip() not in ('', '0', 'false', 'False')
    if debug:
        return CELLS[:2], 1, 4, UNIFIED_POINTS, 0, DEBUG_OUT_DIR, True
    return CELLS, LOW_RATE_CYCLES, HIGH_RATE_CYCLES, UNIFIED_POINTS, \
        FINAL_CYCLE_EXTRA, OUT_DIR, False


def main():
    cells, low_cycles, high_cycles, base_points, final_extra, out_dir, debug = _build_profiles()
    os.makedirs(out_dir, exist_ok=True)
    pybamm.set_logging_level('WARNING')
    # 单线程 BLAS，避免多进程相互抢占线程
    for env_key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(env_key, '1')

    model_options = _model_options()
    worker_count = _worker_count()
    mode_tag = '（debug 冒烟）' if debug else ''
    print(f'[配置{mode_tag}] 模型 DFN + 选项 {model_options}')
    if len(cells) > 1:
        print(f'[配置] 电池 {len(cells)} 只，温度 {cells[0]["temperature_c"]:.1f} ~ '
              f'{cells[-1]["temperature_c"]:.1f} ℃（步长 {TEMP_STEP_C} ℃）')
    else:
        print(f'[配置] 电池 {len(cells)} 只，温度 {cells[0]["temperature_c"]:.1f} ℃')
    print(f'[配置] 每只 {low_cycles} 圈 {LOW_RATE_C_RATE} 化成（不导出）+ '
          f'{high_cycles} 圈 {HIGH_RATE_C_RATE}（导出，含 SEI+裂纹老化）')
    if final_extra > 0:
        print(f'[配置] 每条半循环重采样 {base_points} 点（511/512 奇偶微变，'
              f'末号电池末圈 +{final_extra}）')
    else:
        print(f'[配置] 每条半循环重采样 {base_points} 点（debug）')
    print(f'[配置] var_pts 覆盖 {VAR_PTS_OVERRIDE}；并行进程 {worker_count}；输出 {out_dir}')

    cell_params = []
    for index, cell in enumerate(cells):
        params = dict(cell)
        params.update({
            'cell_index': index,
            'cell_count': len(cells),
            'low_rate_cycles': low_cycles,
            'high_rate_cycles': high_cycles,
            'unified_points': base_points,
            'final_cycle_extra': final_extra,
            'model_options': model_options,
            'debug': debug,
        })
        cell_params.append(params)

    wall_start = time.time()
    results = {}
    failures = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(simulate_cell, params): params
                      for params in cell_params}
        for done, future in enumerate(as_completed(future_map), start=1):
            params = future_map[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - 记录失败电池，最后统一报错
                failures.append(params['test_name'])
                print(f'[失败 {done}/{len(cell_params)}] {params["test_name"]}: {error!r}')
                continue
            results[params['cell_index']] = result
            decay = (1.0 - result['last_charge_cap_mah']
                     / result['first_charge_cap_mah']) * 100.0
            print(f'[完成 {done}/{len(cell_params)}] {result["test_name"]} '
                  f'({high_cycles} 圈, 充电容量 {result["first_charge_cap_mah"]:.3f} -> '
                  f'{result["last_charge_cap_mah"]:.3f} mAh, 衰减 {decay:.1f}%, '
                  f'耗时 {result["elapsed_s"]:.0f} s)')

    if failures:
        raise RuntimeError(f'以下电池仿真失败，请重跑: {failures}')

    cells_runs = [results[index] for index in range(len(cells))]
    cycle_path, capacity_path, cycle_rows, capacity_rows = build_unibo_tables(
        cells_runs, out_dir)
    diag_path, diag_rows = write_coupled_diagnostics(cells_runs, out_dir)

    if not debug:
        soc_path = export_soc_train_csv(
            CELLS[TRAIN_CELL_INDEX]['test_name'],
            cells_runs[TRAIN_CELL_INDEX]['quasi_ocv_discharge'], out_dir)
        table = pd.read_csv(soc_path)
        print('\n[输出] OCV 先验表（000 号电池第 3 圈化成放电，50 点）:', soc_path)
        print(table.to_string(index=False))

    sample = cells_runs[TRAIN_CELL_INDEX]
    soh_points = [(i, value) for i, value in enumerate(sample['soh_series'], start=1)
                  if i == 1 or i % 10 == 0]
    print(f'\n[SOH 曲线] {sample["test_name"]}: ' + ', '.join(
        f'第 {i} 圈 {value:.4f}' for i, value in soh_points))
    print('\n[输出] 统一数据集（UNIBO 同构，SOC/SOH 共用）:')
    print('  ', cycle_path, f'({cycle_rows} 行)')
    print('  ', capacity_path, f'({capacity_rows} 行)')
    print('  ', diag_path, f'({diag_rows} 行，耦合物理场诊断)')

    missing = sorted({column for run in cells_runs
                      for column in run.get('diagnostics_missing', [])})
    if missing:
        print(f'[警告] 诊断列缺失（变量名不匹配，已写 NaN）: {missing}')

    # ---- 容量衰减曲线图（000 号电池，可视化 SEI+裂纹老化）----
    fade_png = os.path.join(out_dir, 'capacity_fade_cell_000.png')
    plot_capacity_fade(sample, fade_png)

    print(f'\n[总计] 总耗时 {(time.time() - wall_start) / 60:.1f} min')


if __name__ == '__main__':
    main()
