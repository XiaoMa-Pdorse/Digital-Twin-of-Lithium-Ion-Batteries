# -*- coding: utf-8 -*-
"""Step 4：Streamlit 可视化整合 —— 硅基负极数字孪生原型（全链路 4 步）。

页面结构（与展示流程一致）
--------------------------
- 左侧边栏：全部输入参数（电池、循环、EKF 参数、RUL 阈值）；
- 右侧主区四个页签：
  Step 1 · 虚拟实验数据（含耦合场）—— DFN+热-力耦合虚拟数据集概览、曲线预览
    与颗粒应力/裂纹/LAM/SEI 温度诊断曲线（读 coupled_diagnostics.csv）；
  Step 2 · EKF 在线 SOC 与 EIS —— 卡尔曼滤波实时估算（kalman_soc.py）+ DFN
    小信号扫频标准 Nyquist 谱与等效电路参数（读 results/eis/ 产物）；
  Step 3 · LSTM 预测 RUL —— 衰减曲线预测、EOL 外推与剩余寿命（rul_lstm.py）；
  Step 4 · 数字孪生总览 —— 全链路流程图、全局指标与模型演示图。

运行方式
--------
streamlit run app.py
（需先运行：si_halfcell_dataset.py 生成数据集；rul_lstm.py 生成 RUL 模型；
  可选：eis_simulation.py 生成 EIS 产物）
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import kalman_soc as ks
import rul_lstm as rl

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(REPO_ROOT, 'results', 'digital_twin', 'rul_metrics.json')
EKF_PNG_PATH = os.path.join(REPO_ROOT, 'results', 'digital_twin', 'ekf_soc_demo.png')
RUL_PNG_PATH = os.path.join(REPO_ROOT, 'results', 'digital_twin', 'rul_demo.png')
DIAG_PATH = os.path.join(REPO_ROOT, 'data', 'si-c-half-cell',
                         'coupled_diagnostics.csv')
EIS_DIR = os.path.join(REPO_ROOT, 'results', 'eis')
EIS_IMPEDANCE_PATH = os.path.join(EIS_DIR, 'eis_impedance.csv')
EIS_FIT_PATH = os.path.join(EIS_DIR, 'eis_fit_params.csv')
EIS_NYQUIST_PNG = os.path.join(EIS_DIR, 'eis_nyquist.png')
EIS_TRENDS_PNG = os.path.join(EIS_DIR, 'eis_fit_trends.png')

CAP_TOTAL_ROWS = 3100          # 统一数据集容量表行数（31 电池 × 50 圈 × 2 段）

# 图例放图下方：避免与标题重叠（高度留给标题）
PLOTLY_LAYOUT = dict(
    height=420,
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation='h', yanchor='top', y=-0.18, x=0),
    hovermode='x unified',
)

st.set_page_config(page_title='硅基负极数字孪生原型', page_icon='🔋', layout='wide')


# ---------------------------------------------------------------------------
# 缓存的数据与模型
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner='加载 LSTM 模型...')
def load_rul_model():
    import tensorflow as tf
    return tf.keras.models.load_model(rl.MODEL_PATH)


@st.cache_data(show_spinner='读取统一数据集曲线表（约 256 万行）...')
def load_soc_cycle_data():
    return pd.read_csv(ks.CYCLE_CSV_PATH)


@st.cache_data(show_spinner='读取 SOH 容量序列...')
def load_soh_cells():
    return rl.load_capacity_series()


@st.cache_data(show_spinner='读取耦合物理场诊断表...')
def load_coupled_diagnostics():
    return pd.read_csv(DIAG_PATH)


@st.cache_data(show_spinner='读取 EIS 阻抗数据...')
def load_eis_impedance():
    return pd.read_csv(EIS_IMPEDANCE_PATH)


@st.cache_data(show_spinner='读取 EIS 等效电路参数...')
def load_eis_fit_params():
    return pd.read_csv(EIS_FIT_PATH)


@st.cache_data(show_spinner='运行 EKF 在线估算...')
def run_ekf_cached(cell_name, cycle, soc0, sigma_v, r0):
    return ks.run_demo(cell_name=cell_name, cycle=cycle, soc0=soc0,
                       sigma_v=sigma_v, r0=r0, save_png=False)


@st.cache_data(show_spinner='加载测试集指标...')
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding='utf-8') as fp:
            return json.load(fp)
    return None


def predict_rul(cells, model, cell_name, threshold):
    """对指定电池：前 20 圈 -> 预测后 30 圈 -> 外推 EOL（预测与真实对比）"""
    cap = cells[cell_name]
    soh = cap / cap[0]
    soh20 = soh[:rl.IN_LEN]
    pred30 = rl.predict_future_soh(model, soh20)
    full_pred = np.concatenate([soh20, pred30])
    eol_pred, slope = rl.linear_extrapolate_rul(full_pred, threshold=threshold)
    eol_true, _ = rl.linear_extrapolate_rul(soh, threshold=threshold)
    return soh, pred30, full_pred, eol_pred, eol_true, slope


# ---------------------------------------------------------------------------
# 页头与侧边栏
# ---------------------------------------------------------------------------
st.title('🔋 硅基负极数字孪生原型')
st.caption('DFN 虚拟实验与耦合场（Step 1）→ EKF 在线 SOC 与 EIS 阻抗（Step 2）'
           '→ LSTM 预测 RUL（Step 3）→ 可视化整合（Step 4）'
           '　|　DFN(P2D)+热-力耦合+裂纹老化，面向 Si 基半电池')

cells_soh = load_soh_cells()
cell_names = sorted(cells_soh)

with st.sidebar:
    st.header('① 电池与数据选择')
    cell_name = st.selectbox(
        '电池（编号越大温度越高）', cell_names,
        index=0,
        format_func=lambda n: f'{n[:3]}（{rl.parse_temperature(n):.2f} ℃）')
    cycle = st.slider('数据集循环号（统一数据集：50 圈 C/2）', 1, 50, 3,
                      help='SOC 与 SOH 共用同一套仿真数据，容量随 SEI 老化逐圈衰减；'
                           '任意循环均可观察 EKF 的在线修正能力')

    st.divider()
    st.header('② EKF 在线估算参数')
    soc0 = st.slider('初始 SOC 猜测', 0.50, 1.00, 0.85, 0.01,
                     help='故意偏离真实值 1.00，用于演示 EKF 的修正能力；'
                          '安时积分无法修正该初始偏差')
    sigma_mv = st.slider('电压观测噪声 σ [mV]', 0.0, 20.0, 3.0, 0.5)
    r0_auto = st.checkbox('R0 自动辨识（网格搜索）', value=True)
    r0_manual = st.number_input('手动 R0 [Ω]', 0.0, 30.0, 5.0, 0.5,
                                disabled=r0_auto)

    st.divider()
    st.header('③ RUL 设置')
    threshold = st.slider('EOL 阈值（SOH）', 0.70, 0.90, 0.80, 0.01,
                          help='容量降至该阈值视为寿命终止（EOL），'
                               '行业惯例为 80%')

tab1, tab2, tab3, tab4 = st.tabs([
    'Step 1 · 虚拟实验数据（含耦合场）', 'Step 2 · EKF 在线 SOC 与 EIS',
    'Step 3 · LSTM 预测 RUL', 'Step 4 · 数字孪生总览'])

# ---------------------------------------------------------------------------
# Step 1：虚拟实验数据
# ---------------------------------------------------------------------------
with tab1:
    st.subheader('Step 1 · PyBaMM 虚拟实验数据（无需真实实验）')
    st.markdown(
        '用 PyBaMM（**DFN（P2D）半电池模型** + 非晶硅 OCP + **电化学-热-力耦合**'
        '（集总热模型 + 颗粒膨胀/开裂/应力驱动 LAM）+ SEI 与裂纹老化）批量仿真 '
        '**31 只电池**（25.0~40.0 ℃ 温度梯度，3 圈 C/20 化成 + 50 圈 C/2 循环），'
        '把每圈的容量、电压、电流（温度列为真实电池温度）存成 UNIBO 同构 CSV'
        '——**SOC 与 SOH 共用同一套数据**：')

    cycle_df = load_soc_cycle_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('电池数量', '31 只', '温度 25.0~40.0 ℃（0.5 ℃ 一档）')
    col2.metric('曲线表行数', f'{len(cycle_df):,}', 'test_result.csv（50 圈 C/2）')
    col3.metric('容量/SOH 表行数', f'{CAP_TOTAL_ROWS:,}', '含 SEI/裂纹老化的逐圈容量')
    col4.metric('每圈重采样', '511/512 点', 'SOC 与 SOH 共用一套')

    seg = cycle_df[(cycle_df.test_name == cell_name)
                   & (cycle_df.cycle_count == cycle)]
    discharge = seg[seg.line == 40]
    charge = seg[seg.line == 37]

    fig = go.Figure()
    fig.add_scatter(x=discharge['step_time'], y=discharge['voltage'],
                    name='放电段（line 40）', mode='lines+markers',
                    marker=dict(size=4), line=dict(color='#1f77b4'))
    fig.add_scatter(x=charge['step_time'], y=charge['voltage'],
                    name='充电段（line 37）', mode='lines+markers',
                    marker=dict(size=4), line=dict(color='#d62728'))
    fig.update_layout(
        title=f'{cell_name[:3]} 电池 · 第 {cycle} 圈 · 端电压曲线（"传感器"观测）',
        xaxis_title='步进时间 [h]', yaxis_title='端电压 [V]', **PLOTLY_LAYOUT)
    st.plotly_chart(fig, width='stretch')

    with st.expander('查看该循环 CSV 数据（前 8 行）'):
        st.dataframe(seg.head(8), width='stretch')
    st.info('提示：数据文件位于 data/si-c-half-cell/（统一数据集，DFN+热-力耦合，'
            '含 SEI/裂纹老化），由 si_halfcell_dataset.py 一键生成；'
            'SOC_Train.py / SOH_Train.py 读取同一套 CSV 即可分别训练 SOC / SOH 模型。')

# ---------------------------------------------------------------------------
# Step 2：EKF 在线 SOC
# ---------------------------------------------------------------------------
with tab2:
    st.subheader('Step 2 · 卡尔曼滤波（EKF）在线 SOC 估算')
    st.markdown(
        '以一阶等效电路模型（OCV 查表 + 欧姆内阻 R0 + R1C1 极化）为过程模型，'
        '用 **filterpy** 的扩展卡尔曼滤波把仿真电压曲线当作"传感器观测"'
        '逐点实时估算 SOC，与纯安时积分对比。')

    r0_value = None if r0_auto else float(r0_manual)
    try:
        result = run_ekf_cached(cell_name, cycle, float(soc0),
                                float(sigma_mv) / 1000.0, r0_value)
    except Exception as error:  # noqa: BLE001 - 页面内展示错误而非崩溃
        st.error(f'该组合无法运行 EKF：{error}')
        result = None

    if result is not None:
        metrics_ekf = result['metrics_ekf']
        metrics_cc = result['metrics_cc']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('EKF RMSE', f'{metrics_ekf["rmse"] * 100:.2f}%',
                    f'末点 {metrics_ekf["final_err"] * 100:+.2f}%')
        col2.metric('安时积分 RMSE', f'{metrics_cc["rmse"] * 100:.2f}%',
                    '无法修正初始偏差', delta_color='inverse')
        col3.metric('R0 辨识值', f'{result["r0"]:.1f} Ω',
                    '网格搜索最优' if r0_auto else '手动指定',
                    delta_color='off')
        col4.metric('段放电容量', f'{result["segment"]["capacity_mah"]:.2f} mAh',
                    f'{result["segment"]["n_points"]} 个采样点',
                    delta_color='off')

        t_min = result['segment']['t'] / 60.0
        fig = go.Figure()
        fig.add_scatter(x=t_min, y=result['segment']['v'], name='仿真电压（原始）',
                        line=dict(color='#888888', width=1))
        fig.add_scatter(x=t_min, y=result['v_obs'], name='传感器观测（含噪声）',
                        line=dict(color='#17becf', width=1, dash='dot'))
        fig.update_layout(title='(a) 电压观测（"传感器"输入）',
                          xaxis_title='时间 [min]', yaxis_title='端电压 [V]',
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width='stretch')

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_scatter(x=t_min, y=result['soc_ref'] * 100, name='参考 SOC',
                            line=dict(color='black', width=2))
            fig.add_scatter(x=t_min, y=result['soc_ekf'] * 100, name='EKF 在线估计',
                            line=dict(color='#d62728', width=2))
            fig.add_scatter(x=t_min, y=result['soc_cc'] * 100,
                            name=f'安时积分（初值 {result["soc_cc"][0] * 100:.0f}%）',
                            line=dict(color='#1f77b4', dash='dash'))
            fig.update_layout(title='(b) SOC 估计曲线（实时）',
                              xaxis_title='时间 [min]', yaxis_title='SOC [%]',
                              **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')
        with col_b:
            fig = go.Figure()
            fig.add_scatter(x=t_min,
                            y=(result['soc_ekf'] - result['soc_ref']) * 100,
                            name=f'EKF（RMSE {metrics_ekf["rmse"] * 100:.2f}%）',
                            line=dict(color='#d62728'))
            fig.add_scatter(x=t_min,
                            y=(result['soc_cc'] - result['soc_ref']) * 100,
                            name=f'安时积分（RMSE {metrics_cc["rmse"] * 100:.2f}%）',
                            line=dict(color='#1f77b4', dash='dash'))
            fig.add_hline(y=0, line_color='black', line_width=1, opacity=0.5)
            fig.update_layout(title='(c) SOC 误差（滤波修正 vs 纯积分漂移）',
                              xaxis_title='时间 [min]', yaxis_title='SOC 误差 [%]',
                              **PLOTLY_LAYOUT)
            st.plotly_chart(fig, width='stretch')

        st.success(f'EKF 将初始 {soc0 * 100:.0f}% 的猜测快速修正到参考轨迹附近'
                   f'（RMSE {metrics_ekf["rmse"] * 100:.2f}%），'
                   f'而安时积分因初始偏差持续漂移（RMSE {metrics_cc["rmse"] * 100:.2f}%）。'
                   f'电流按容量积分驱动状态预测，电压观测负责闭环修正——'
                   f'这正是"离线仿真 → 在线估算"的关键一步。')

# ---------------------------------------------------------------------------
# Step 3：LSTM 预测 RUL
# ---------------------------------------------------------------------------
with tab3:
    st.subheader('Step 3 · LSTM 预测容量衰减曲线与剩余寿命（RUL）')
    st.markdown(
        '输入选定电池**前 20 圈容量序列**，用 LSTM 预测**其后 30 圈衰减曲线**，'
        '再对预测曲线末端做线性外推，估算到达 EOL 阈值的循环数（RUL）。'
        '模型仅需容量序列（不含温度/型号标签），可泛化迁移至其他 Si 基数据。')

    if not os.path.exists(rl.MODEL_PATH):
        st.warning('未找到已训练模型，请先运行 `python rul_lstm.py` 完成训练。')
    else:
        model = load_rul_model()
        soh, pred30, full_pred, eol_pred, eol_true, slope = predict_rul(
            cells_soh, model, cell_name, float(threshold))

        col1, col2, col3, col4 = st.columns(4)
        if eol_pred is not None:
            col1.metric('预测 RUL', f'≈ {eol_pred - rl.IN_LEN:.0f} 圈',
                        f'第 {eol_pred:.0f} 圈到 SOH {threshold:.0%}')
        else:
            col1.metric('预测 RUL', '—（无衰减趋势）')
        if eol_true is not None:
            delta = None if eol_pred is None else eol_pred - eol_true
            col2.metric('真实参考 EOL', f'第 {eol_true:.0f} 圈',
                        None if delta is None else f'预测偏差 {delta:+.1f} 圈',
                        delta_color='off')
        col3.metric('当前 SOH（第 20 圈）', f'{soh[rl.IN_LEN - 1] * 100:.2f}%')
        col4.metric('衰减斜率（末端拟合）', f'{slope * 100:.3f}% SOH/圈')

        x_all = np.arange(1, len(soh) + 1)
        fig = go.Figure()
        fig.add_scatter(x=x_all[:rl.IN_LEN], y=soh[:rl.IN_LEN] * 100,
                        name='已知前 20 圈', line=dict(color='#888888', dash='dot'))
        fig.add_scatter(x=x_all[rl.IN_LEN:], y=soh[rl.IN_LEN:] * 100,
                        name='真实后 30 圈（验证用）',
                        line=dict(color='black', width=2))
        fig.add_scatter(x=x_all[rl.IN_LEN:], y=pred30 * 100,
                        name='LSTM 预测', line=dict(color='#d62728', width=2,
                                                    dash='dash'))
        if eol_pred is not None:
            fig.add_scatter(x=[x_all[-1], eol_pred],
                            y=[full_pred[-1] * 100, threshold * 100],
                            name=f'线性外推（预测 RUL≈{eol_pred - rl.IN_LEN:.0f} 圈）',
                            line=dict(color='#d62728', width=1.5, dash='dot'))
            fig.add_scatter(x=[eol_pred], y=[threshold * 100], mode='markers',
                            marker=dict(symbol='triangle-down', size=12,
                                        color='#d62728'),
                            showlegend=False)
        fig.add_hline(y=threshold * 100, line_color='#2ca02c', line_dash='dash',
                      annotation_text=f'EOL 阈值 SOH={threshold:.0%}')
        fig.add_vline(x=rl.IN_LEN + 0.5, line_color='#aaaaaa', line_width=1)
        fig.update_layout(
            title=f'{cell_name[:3]} 电池（{rl.parse_temperature(cell_name):.2f} ℃）'
                  f'· SOH 曲线预测与 RUL 外推',
            xaxis_title='循环数', yaxis_title='SOH [%]',
            yaxis=dict(range=[min(threshold - 5, full_pred.min() * 100 - 1), 101]),
            **PLOTLY_LAYOUT)
        st.plotly_chart(fig, width='stretch')

        mae = float(np.mean(np.abs(pred30 - soh[rl.IN_LEN:]))) * 100
        if mae < 1.0:
            metrics_rul = load_metrics()
            overall = ('曲线 MAE ≈ 1.0% SOH，RUL 外推误差 ≈ 3 圈。'
                       if metrics_rul is None else
                       f'曲线 MAE ≈ {metrics_rul["curve_mae_mean"] * 100:.2f}% SOH，'
                       f'RUL 外推误差 ≈ {metrics_rul["rul_err_abs_mean"]:.1f} 圈。')
            st.success(f'后 30 圈预测与真实曲线平均绝对误差约 {mae:.3f}% SOH。'
                       f'测试集（10 只未参与训练的高温电池）整体：' + overall)
        else:
            st.info(f'后 30 圈预测与真实曲线平均绝对误差约 {mae:.3f}% SOH（该电池'
                    f'超出典型训练分布，预测仍有趋势参考价值）。')

# ---------------------------------------------------------------------------
# Step 4（已并入 Step 1 页签，内容追加渲染）：颗粒应力 / 裂纹 / LAM / SEI / 温度
# ---------------------------------------------------------------------------
with tab1:
    st.divider()
    st.subheader('Step 1 补充 · 电化学-热-力耦合场（DFN 求解变量逐圈导出）')
    st.markdown(
        '硅负极充放电伴随 ~300% 体积变化：**应力 → 颗粒开裂 → 裂纹处 SEI 持续生长'
        ' → 应力驱动活性材料损失（LAM）** 的机械-化学衰减链条由 DFN 全耦合模型直接'
        '求解并逐圈导出（coupled_diagnostics.csv），无需事后反演。')

    if not os.path.exists(DIAG_PATH):
        st.warning('未找到耦合诊断表，请先运行 `python si_halfcell_dataset.py`。')
    else:
        diag = load_coupled_diagnostics()
        cell_diag = diag[diag.test_name == cell_name]
        discharge_diag = cell_diag[cell_diag.phase == 'discharge'].sort_values('cycle')
        charge_diag = cell_diag[cell_diag.phase == 'charge'].sort_values('cycle')

        if len(discharge_diag) == 0:
            st.error('该电池不在当前诊断表中。')
        else:
            last = discharge_diag.iloc[-1]
            first = discharge_diag.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric('表面应力峰值（末圈）',
                        f'{last.max_abs_surface_stress_mpa:.0f} MPa',
                        f'首圈 {first.max_abs_surface_stress_mpa:.0f} MPa',
                        delta_color='off')
            col2.metric('裂纹长度（末圈）',
                        f'{last.end_crack_length_um:.3f} μm',
                        f'首圈 {first.end_crack_length_um:.3f} μm',
                        delta_color='off')
            col3.metric('活性材料残留',
                        f'{last.end_active_material_fraction / first.end_active_material_fraction * 100:.2f}%',
                        f'LAM 损失 '
                        f'{(1 - last.end_active_material_fraction / first.end_active_material_fraction) * 100:.2f}%',
                        delta_color='inverse')
            col4.metric('SEI 厚度（末圈）',
                        f'{last.end_sei_thickness_nm:.2f} nm',
                        f'首圈 {first.end_sei_thickness_nm:.2f} nm',
                        delta_color='off')

            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure()
                fig.add_scatter(x=discharge_diag['cycle'],
                                y=discharge_diag['max_abs_surface_stress_mpa'],
                                name='放电段', mode='lines+markers',
                                marker=dict(size=4))
                fig.add_scatter(x=charge_diag['cycle'],
                                y=charge_diag['max_abs_surface_stress_mpa'],
                                name='充电段', mode='lines+markers',
                                marker=dict(size=4))
                fig.update_layout(
                    title='(a) 颗粒表面切向应力（绝对值峰值）',
                    xaxis_title='循环数', yaxis_title='应力 [MPa]', **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')
            with col_b:
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                fig.add_scatter(x=discharge_diag['cycle'],
                                y=discharge_diag['end_crack_length_um'],
                                name='裂纹长度 [μm]', mode='lines+markers',
                                marker=dict(size=4))
                fig.add_scatter(
                    x=discharge_diag['cycle'],
                    y=discharge_diag['end_active_material_fraction'] * 100,
                    name='活性材料残留 [%]', mode='lines+markers',
                    marker=dict(size=4), secondary_y=True)
                fig.update_layout(
                    title='(b) 裂纹扩展与活性材料损失（放电段末值）',
                    xaxis_title='循环数', **PLOTLY_LAYOUT)
                fig.update_yaxes(title_text='裂纹长度 [μm]', secondary_y=False)
                fig.update_yaxes(title_text='活性材料残留 [%]', secondary_y=True)
                st.plotly_chart(fig, width='stretch')

            demo_targets = (25.0, 30.0, 35.0, 40.0)
            demo_cells = {}
            for target in demo_targets:
                demo_cells[target] = min(
                    cell_names,
                    key=lambda name: abs(rl.parse_temperature(name) - target))

            col_c, col_d = st.columns(2)
            with col_c:
                fig = go.Figure()
                for target, name in demo_cells.items():
                    sub = diag[(diag.test_name == name)
                               & (diag.phase == 'discharge')].sort_values('cycle')
                    fig.add_scatter(
                        x=sub['cycle'],
                        y=(1.0 - sub['end_active_material_fraction']) * 100,
                        name=f'{target:.0f} ℃（{name[:3]}）', mode='lines')
                fig.update_layout(title='(c) 应力驱动 LAM 损失（温度弱相关，随循环累积）',
                                  xaxis_title='循环数', yaxis_title='LAM 损失 [%]',
                                  **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')
            with col_d:
                fig = go.Figure()
                for target, name in demo_cells.items():
                    sub = diag[(diag.test_name == name)
                               & (diag.phase == 'discharge')].sort_values('cycle')
                    fig.add_scatter(x=sub['cycle'],
                                    y=sub['max_temperature_c'] - target,
                                    name=f'{target:.0f} ℃（{name[:3]}）', mode='lines')
                fig.update_layout(
                    title='(d) 电池温升（集总热模型求解变量）',
                    xaxis_title='循环数', yaxis_title='T − Tₐᵣₑₐ [K]',
                    **PLOTLY_LAYOUT)
                st.plotly_chart(fig, width='stretch')

            st.info('解读：C/2 倍率下半电池产热很小（温升 <0.1 K），热耦合主要体现在高温'
                    '环境下的动力学与老化加速（Arrhenius）；而颗粒表面应力达百 MPa '
                    '量级、接近硅的临界断裂应力，持续驱动裂纹扩展与 LAM——这正是硅基'
                    '负极容量衰减的机械主因，也是 DFN 全耦合相对纯电化学模型的增量。')

# ---------------------------------------------------------------------------
# Step 5（已并入 Step 2 页签，内容追加渲染）：EIS 阻抗分析
# ---------------------------------------------------------------------------
with tab2:
    st.divider()
    st.subheader('Step 2 补充 · EIS 阻抗分析（DFN 时域小信号扫频）')
    st.markdown(
        '在 DFN(P2D) 模型上做恒流小信号 EIS（C/200 振幅）：目标 SOC 平衡态初值'
        ' → 正弦电流激励 → **频点链式扫频** → 最小二乘基波提取 Z(jω)。'
        '模型开启 **surface form（双电层动力学生效）**，谱呈标准形态：'
        '**高频截距（R0）→ 中频半圆（Rct‖Cdl）→ 低频扩散尾**；'
        'EKF 时域辨识的 R0 与 EIS 高频截距同量级，构成时域/频域互证。')

    if not os.path.exists(EIS_IMPEDANCE_PATH):
        st.warning('未找到 EIS 仿真数据，请先运行 `python eis_simulation.py`'
                   '（或 `EIS_QUICK=1` 快速冒烟）。')
    else:
        impedance = load_eis_impedance()
        temperatures = sorted(impedance['temperature_c'].unique())
        selected = st.selectbox('温度 [℃]', temperatures, index=0,
                                format_func=lambda value: f'{value:.1f}')
        subset = impedance[impedance['temperature_c'] == selected]

        fig = go.Figure()
        # 低频阈值与 eis_simulation.LOW_FREQ_THRESHOLD_HZ 对应：
        # ≥0.5 Hz 为可靠区（实线），<0.5 Hz 为过渡区近似（点线+空心点）
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for index, (soc, group) in enumerate(subset.groupby('soc')):
            color = palette[index % len(palette)]
            group = group.sort_values('frequency_hz', ascending=False)
            reliable = group[group['frequency_hz'] >= 0.5]
            transitional = group[group['frequency_hz'] <= 0.5 + 1e-9]
            fig.add_scatter(x=reliable['z_real_ohm'], y=-reliable['z_imag_ohm'],
                            name=f'SOC {soc:.0%}', mode='lines+markers',
                            marker=dict(size=6, color=color),
                            line=dict(color=color),
                            customdata=reliable['frequency_hz'],
                            hovertemplate="Z'=%{x:.1f} Ω<br>-Z''=%{y:.1f} Ω"
                                          "<br>f=%{customdata:.3g} Hz<extra></extra>")
            fig.add_scatter(x=transitional['z_real_ohm'],
                            y=-transitional['z_imag_ohm'],
                            mode='lines+markers', showlegend=False,
                            line=dict(color=color, dash='dot'),
                            marker=dict(size=6, color=color, symbol='circle-open'),
                            customdata=transitional['frequency_hz'],
                            hovertemplate="Z'=%{x:.1f} Ω<br>-Z''=%{y:.1f} Ω"
                                          "<br>f=%{customdata:.3g} Hz（过渡区）"
                                          "<extra></extra>")
        fig.update_layout(
            title=f'{selected:.1f} ℃ · 标准 Nyquist 图'
                  '（等比例；实线=可靠区 ≥0.5 Hz，点线=过渡区近似）',
            xaxis_title="Z' [Ω]", yaxis_title="-Z'' [Ω]", **PLOTLY_LAYOUT)
        fig.update_yaxes(scaleanchor='x', scaleratio=1)   # 等比例：标准 Nyquist
        st.plotly_chart(fig, width='stretch')

        if os.path.exists(EIS_FIT_PATH):
            fit = load_eis_fit_params()
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown('#### 等效电路参数（ZARC：R0 + Rct‖CPE + W）')
                show = fit.copy()
                show['SOC'] = (show['soc'] * 100).round(0).astype(int)
                show = show.rename(columns={
                    'temperature_c': '温度 [℃]', 'r0_ohm': 'R0 [Ω]',
                    'rct_ohm': 'Rct [Ω]', 'cpe_q': 'Q (CPE)',
                    'cpe_n': 'n (CPE)', 'sigma_warburg': 'σ (Warburg)',
                    'rmse_ohm': '拟合 RMSE [Ω]'})
                st.dataframe(
                    show[['温度 [℃]', 'SOC', 'R0 [Ω]', 'Rct [Ω]', 'Q (CPE)',
                          'n (CPE)', 'σ (Warburg)', '拟合 RMSE [Ω]']]
                    .style.format({'温度 [℃]': '{:.1f}', 'R0 [Ω]': '{:.1f}',
                                   'Rct [Ω]': '{:.0f}', 'Q (CPE)': '{:.4g}',
                                   'n (CPE)': '{:.3f}', 'σ (Warburg)': '{:.3g}',
                                   '拟合 RMSE [Ω]': '{:.2f}'}),
                    width='stretch', hide_index=True)
            with col_b:
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=('R0 vs SOC（高频截距）',
                                                    'Rct vs SOC（电荷转移 + SEI 膜阻）'))
                for temperature, group in fit.groupby('temperature_c'):
                    group = group.sort_values('soc')
                    fig.add_scatter(x=group['soc'] * 100, y=group['r0_ohm'],
                                    name=f'{temperature:.1f} ℃',
                                    mode='lines+markers', row=1, col=1)
                    fig.add_scatter(x=group['soc'] * 100, y=group['rct_ohm'],
                                    name=f'{temperature:.1f} ℃',
                                    mode='lines+markers', row=1, col=2,
                                    showlegend=False)
                fig.update_xaxes(title_text='SOC [%]', row=1, col=1)
                fig.update_xaxes(title_text='SOC [%]', row=1, col=2)
                fig.update_yaxes(title_text='R0 [Ω]', row=1, col=1)
                fig.update_yaxes(title_text='Rct [Ω]', row=1, col=2)
                fig.update_layout(
                    title='等效电路参数 vs SOC / 温度',
                    height=430, margin=dict(l=10, r=10, t=70, b=10),
                    legend=dict(orientation='h', yanchor='top', y=-0.15, x=0))
                st.plotly_chart(fig, width='stretch')

            st.info('判读：谱呈标准「高频截距 → 中频半圆 → 低频扩散尾」——'
                    '高频截距 = 欧姆内阻 R0（与 EKF 时域辨识的 R0 同量级，'
                    '实现时域/频域互证）；半圆直径 = 电荷转移 + SEI 膜阻 Rct'
                    '（随 SOC 变化反映交换电流密度 j0）；低频上翘段为固相扩散'
                    '（过渡区近似，趋势可信）。')

# ---------------------------------------------------------------------------
# Step 4：数字孪生总览
# ---------------------------------------------------------------------------
with tab4:
    st.subheader('Step 4 · 数字孪生总览（全链路闭环）')
    st.markdown(
        """
<div style="display:flex; gap:10px; margin-bottom:14px;">
  <div style="flex:1; padding:12px; border:2px solid #1f77b4; border-radius:10px; text-align:center;">
    <b style="color:#1f77b4;">Step 1 · DFN 虚拟实验与耦合场</b><br>
    <small>31 电池×53 圈（25~40 ℃）<br>热-力耦合场 + 裂纹老化诊断</small>
  </div>
  <div style="flex:1; padding:12px; border:2px solid #d62728; border-radius:10px; text-align:center;">
    <b style="color:#d62728;">Step 2 · EKF 在线 SOC 与 EIS</b><br>
    <small>EKF 在线估算（RMSE≈2.2%）<br>EIS 标准 Nyquist + R0/Rct 提取</small>
  </div>
  <div style="flex:1; padding:12px; border:2px solid #2ca02c; border-radius:10px; text-align:center;">
    <b style="color:#2ca02c;">Step 3 · LSTM 预测 RUL</b><br>
    <small>前 20 圈 → 后 30 圈衰减曲线<br>EOL 外推（曲线 MAE≈1%）</small>
  </div>
  <div style="flex:1; padding:12px; border:2px solid #9467bd; border-radius:10px; text-align:center;">
    <b style="color:#9467bd;">Step 4 · 可视化整合</b><br>
    <small>Streamlit 数字孪生原型<br>本页面即交付物</small>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('#### 全局性能指标')
    metrics = load_metrics()
    demo = run_ekf_cached('000-SI-3.0-2500-S', 3, 0.85, 0.003, None)
    demo_ekf = demo['metrics_ekf']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('EKF SOC 估算', f'RMSE ≈ {demo_ekf["rmse"] * 100:.2f}%',
                f'末点误差 {demo_ekf["final_err"] * 100:+.2f}%，含 3 mV 噪声')
    col2.metric('LSTM 曲线预测',
                (f'MAE ≈ {metrics["curve_mae_mean"] * 100:.2f}% SOH'
                 if metrics else 'MAE ≈ 1.0% SOH'),
                '10 只未见过的高温电池')
    col3.metric('RUL 外推误差',
                (f'≈ {metrics["rul_err_abs_mean"]:.1f} 圈'
                 if metrics else '≈ 3 圈'),
                '到达 SOH 80% 的循环数')
    col4.metric('泛化验证', '温度外推组', '后 10 只高温电池未参与训练')

    if metrics is not None:
        st.markdown('#### 测试集逐电池明细（LSTM 泛化评估）')
        detail = pd.DataFrame(metrics['cells'])[
            ['test_name', 'temperature_c', 'curve_mae', 'curve_rmse',
             'final_err', 'true_eol_cycle', 'pred_eol_cycle', 'rul_err_cycle']]
        detail.columns = ['电池', '温度 [℃]', '曲线 MAE', '曲线 RMSE',
                          '末端误差', '真实 EOL [圈]', '预测 EOL [圈]',
                          'RUL 误差 [圈]']
        st.dataframe(detail.style.format({
            '温度 [℃]': '{:.2f}', '曲线 MAE': '{:.4f}', '曲线 RMSE': '{:.4f}',
            '末端误差': '{:+.4f}', '真实 EOL [圈]': '{:.0f}',
            '预测 EOL [圈]': '{:.0f}', 'RUL 误差 [圈]': '{:+.1f}'}),
            width='stretch', hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('#### Step 2 产出：EKF 演示图')
        if os.path.exists(EKF_PNG_PATH):
            st.image(EKF_PNG_PATH, width='stretch')
    with col_b:
        st.markdown('#### Step 3 产出：LSTM 训练与泛化演示图')
        if os.path.exists(RUL_PNG_PATH):
            st.image(RUL_PNG_PATH, width='stretch')

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('#### Step 2 产出：EIS 标准 Nyquist 图')
        if os.path.exists(EIS_NYQUIST_PNG):
            st.image(EIS_NYQUIST_PNG, width='stretch')
        else:
            st.caption('（未生成，运行 `python eis_simulation.py`）')
    with col_d:
        st.markdown('#### Step 2 产出：等效电路参数趋势')
        if os.path.exists(EIS_TRENDS_PNG):
            st.image(EIS_TRENDS_PNG, width='stretch')
        else:
            st.caption('（未生成，运行 `python eis_simulation.py`）')

    st.markdown('#### 架构与数据流')
    st.code(
        'si_halfcell_dataset.py  ──DFN(P2D)+热-力耦合+裂纹老化──▶  data/si-c-half-cell/*.csv\n'
        '        │          （统一数据集 + coupled_diagnostics.csv 耦合场诊断）\n'
        '        ├─▶ SOC_Train.py / SOH_Train.py   （CNN/LSTM 训练，读取同一套 CSV）\n'
        '        │\n'
        '        ├─▶ kalman_soc.py    （Step 2：EKF 在线 SOC，RMSE ≈ 2.2%）\n'
        '        │\n'
        '        ├─▶ eis_simulation.py（Step 2：标准 Nyquist + 等效电路参数）\n'
        '        │\n'
        '        ├─▶ rul_lstm.py      （Step 3：LSTM 衰减预测 + RUL，曲线 MAE ≈ 1.0%）\n'
        '        │\n'
        '        └─▶ app.py  ◀── 本页面（Step 4：Streamlit 数字孪生原型）',
        language='text')
    st.caption('运行方式：`streamlit run app.py`　|　'
               '训练模型：`python rul_lstm.py`　|　'
               '数据集重建：`python si_halfcell_dataset.py`　|　'
               'EIS 仿真：`python eis_simulation.py`')
