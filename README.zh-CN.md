# 硅基负极电池数字孪生原型

**DFN（P2D）虚拟实验 → EKF 在线 SOC → EIS 阻抗分析 → LSTM 寿命预测 → Streamlit 可视化**

[English](README.md)

面向硅基（非晶硅）半电池的**机理-数据混合**数字孪生原型：用 PyBaMM 的 **DFN（真 P2D）**
全阶电化学模型耦合**热-力老化**（集总热 + 颗粒膨胀开裂 + 裂纹处 SEI + 应力驱动 LAM），
批量生成 31 只电池 × 25–40 ℃ 的虚拟数据集；在此基础上构建在线估计链路——
**EKF** 在线估算 SOC、**EIS** 频域阻抗表征、**LSTM** 预测容量衰减与 RUL，
并以 **Streamlit** 交互式仪表盘整合展示全链路。

---

## 项目亮点

- **机理深度——真 P2D + 全耦合老化**：完整 DFN（固相/液相扩散 PDE + Butler-Volmer 动力学），
  而非 SPMe 的液相均匀近似；叠加 `thermal='lumped'` 电化学-热耦合、
  `particle mechanics='swelling and cracking'` 颗粒膨胀开裂、
  `SEI on cracks` 裂纹处 SEI 持续生长与 `stress-driven LAM` 应力驱动活性材料损失——
  完整打通硅负极「**应力 → 裂纹 → 裂纹处 SEI → 容量衰减**」链条。
- **可追溯的虚拟数据集**：31 只电池（25.0–40.0 ℃，0.5 ℃ 一档）× 53 圈
  （3 圈 C/20 化成 + 50 圈 C/2），导出**真实电池温度**（求解变量）与
  `coupled_diagnostics.csv` 逐圈物理场诊断（颗粒应力/裂纹长度/SEI 厚度/LAM）。
- **频域表征——EIS 阻抗仿真**：DFN 时域小信号正弦扫频（1000 Hz → 0.05 Hz），
  通过 `'surface form': 'differential'` 激活双电层动力学，得到教科书级
  「高频截距 → 中频半圆 → 低频扩散尾」标准 Nyquist 谱，并完成
  ZARC 等效电路拟合（拟合 RMSE ≈ 1 Ω）。
- **在线估计 + 数据驱动预测**：`filterpy` EKF（OCV 查表 + R0/R1C1 等效电路，
  3 mV 噪声下 SOC RMSE ≈ 2.2%）与小型 LSTM（前 20 圈 SOH → 后 30 圈，曲线
  MAE ≈ 1.0% SOH、未见过高温电池 RUL 误差 ≈ 3.2 圈）。
- **交互式仪表盘**：Streamlit 4 页签——虚拟实验与耦合场、EKF 与 EIS、
  LSTM 寿命预测、全链路总览。

## 全链路流程

```
┌──────────────────────────────────────────────────────────────────────┐
│  si_halfcell_dataset.py   (PyBaMM 26.x)                              │
│  DFN (P2D) + 集总热 + 颗粒力学 + 裂纹 SEI + 应力驱动 LAM                │
│  31 只电池 × (3 圈化成 + 50 圈 C/2)  →  UNIBO 同构 CSV                 │
│  + coupled_diagnostics.csv（耦合场诊断） + OCV-SOC 先验表              │
└──────────────┬───────────────────────────────────────────────────────┘
               │
     ┌─────────┼────────────────┬──────────────────────┐
     ▼         ▼                ▼                      ▼
 SOC_Train.py  kalman_soc.py  eis_simulation.py      rul_lstm.py
 LSTM          EKF 在线 SOC    小信号 EIS 扫频         LSTM 寿命预测
               (RMSE≈2.2%)     Nyquist + ZARC 拟合    (MAE≈1%, ±3 圈)
     │              │            │                      │
     └──────────────┴────────────┴──────────────────────┘
                              ▼
                     app.py  (Streamlit 数字孪生, 4 页签)
```

## 目录结构

```
├── si_halfcell_dataset.py   # 一键生成 DFN 全耦合虚拟数据集
├── eis_simulation.py        # 时域小信号 EIS → 标准 Nyquist + ZARC 拟合
├── SOC_Train.py             # LSTM SOC 训练
├── SOH_Train.py             # CNN SOH 训练
├── kalman_soc.py            # Step 2：EKF 在线 SOC 估算（含演示图）
├── rul_lstm.py              # Step 3：LSTM 容量衰减 / RUL 预测
├── app.py                   # Step 4：Streamlit 数字孪生仪表盘
├── Gemini_SOC_Train.py      # 旧版 SOC 变体（基于原始 UNIBO 数据集）
├── data_processing/         # 数据加载与归一化（UNIBO 兼容）
├── data/                    # 数据集（按需生成，见「快速开始」）
├── results/                 # 演示图、指标 JSON、EIS 分析数据表
├── Picture/                 # 网络结构图
└── requirements.txt
```

## 快速开始

### 0. 环境安装

```bash
pip install -r requirements.txt
# Python 3.12 · PyBaMM 26.4.1 · TensorFlow 2.16 / Keras 3 · Streamlit 1.63
```

### 1. 生成虚拟数据集（必需，12 进程并行约 10–40 分钟）

```bash
python si_halfcell_dataset.py
```

输出到 `data/si-c-half-cell/`：`test_result.csv`（约 159 万行曲线表）、
`test_result_trial_end.csv`（容量/SOH 表 3100 行）、
`coupled_diagnostics.csv`（逐圈应力/裂纹/LAM/SEI/温度诊断）、
`si_c_half_cell_soc_train.csv`（OCV-SOC 先验表）与容量衰减图。

建议先用冒烟模式验证环境（2 只电池 5 圈，约 30 秒）：

```powershell
$env:SI_HALFCELL_DEBUG='1'; python si_halfcell_dataset.py
```

### 2. 准备仪表盘依赖的模型产物

```bash
python kalman_soc.py     # EKF 演示 → results/digital_twin/ekf_soc_demo.png
python rul_lstm.py       # LSTM 训练 + 指标 → results/digital_twin/
```

### 3. 可选：训练 SOC / SOH 网络

```bash
python SOC_Train.py      # → …_lstm_soc_percentage.keras
python SOH_Train.py      # → …_cnn_soh_percentage.keras
```

### 4. 可选：EIS 阻抗仿真（4 温度 × 5 SOC × 13 频点，约 8 分钟）

```bash
python eis_simulation.py            # 全量扫频 → results/eis/
# $env:EIS_QUICK='1'; python eis_simulation.py    # 冒烟（约 1 分钟）
```

### 5. 启动数字孪生仪表盘

```bash
streamlit run app.py
```

页签：**① 虚拟实验数据与耦合场 · ② EKF 在线 SOC 与 EIS · ③ LSTM 预测 RUL · ④ 全链路总览**

## 数据集规格

| 项目 | 取值 |
|---|---|
| 体系 | 非晶硅工作电极 / 锂金属对电极（半电池） |
| 模型 | DFN(P2D) + 集总热 + 膨胀开裂力学 + 裂纹 SEI + 应力驱动 LAM |
| 电池 | 31 只（编号 000–030），环境温度 25.0–40.0 ℃，步长 0.5 ℃ |
| 协议 | 3 圈 C/20 化成（不导出）+ 50 圈 C/2 充放电 |
| 重采样 | 每条半循环 511/512 点（按容量网格，供 CNN/LSTM 滑窗） |
| 容量衰减（50 圈） | 25 ℃ 19.8% → 40 ℃ 25.5%（SEI 主导，Arrhenius 加速） |
| 附属表 | SOH 容量表、逐圈耦合场诊断表、OCV-SOC 先验表 |

## 实测指标

| 指标 | 数值 |
|---|---|
| EKF 在线 SOC（C/2、3 mV 噪声、初值偏差 15%） | **RMSE 2.22%**（安时积分 14.2%） |
| LSTM RUL（10 只未见过的高温电池） | 曲线 **MAE 1.0% SOH**，RUL 误差 **≈ 3.2 圈** |
| EIS（DFN，1000→0.05 Hz） | R0 ≈ **11.5 Ω**；Rct 70–212 Ω（随温降、随 SOC 呈 U 形）；ZARC 拟合 RMSE ≈ **1 Ω** |
| EIS ↔ EKF 互证 | 频域高频截距 R0 ≈ 11.5 Ω，与时域 EKF 辨识 17.5 Ω 同量级 |

## 数据来源与参考文献

- **UNIBO Powertools 数据集**（保留原始加载器兼容）：从 Mendeley `n6xg5fzsbv/1`
  下载解压到 `data/unibo-powertools-dataset/`，详见 `data/README.md`。
- **PyBaMM** — *Python Battery Mathematical Modelling*, JORS 9(1):14, 2021.
- **O'Kane et al.**, *Lithium-ion battery degradation: how to model it.*
  Phys. Chem. Chem. Phys. 24:7909–7922, 2022（OKane2022 参数集）。
- **Chen et al.**, *Development of Experimental Techniques for Parameterization of
  Multi-scale Lithium-ion Battery Models.* JES 167(8):080534, 2020（Chen2020 参数）。
- **Wong et al.**, *Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at
  Various Battery Specifications and Discharge Cycles.* GoodIT '21
  （UNIBO 兼容训练管线的原始架构）。

## 致谢

本项目基于 PyBaMM 开源生态与 Kei Long Wong 等人的电池状态估计代码库扩展而来：
DFN 全耦合老化工作流、虚拟数据集、EIS 仿真与数字孪生仪表盘均为在其之上的增量开发。
仅供研究与学习使用——参数集与虚拟数据不代表任何真实电芯。
