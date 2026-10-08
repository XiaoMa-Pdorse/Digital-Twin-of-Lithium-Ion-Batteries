# Silicon-Anode Battery Digital Twin

**DFN (P2D) virtual experiments → EKF online SOC → EIS impedance → LSTM RUL → Streamlit dashboard**

[中文说明](README.zh-CN.md)

A physics-grounded battery digital-twin prototype for silicon-anode half-cells.
PyBaMM **DFN (true P2D)** simulations with coupled **electrochemical–thermal–mechanical
aging** generate a 31-cell / 25–40 °C virtual dataset, which then feeds an online
estimation stack: **EKF** state-of-charge, **EIS** impedance analysis, **LSTM**
remaining-useful-life prediction, and an interactive **Streamlit** dashboard.

---

## Highlights

- **Physics depth — real P2D + coupled aging.** Full-order DFN (solid/liquid-phase
  diffusion + Butler–Volmer kinetics) instead of SPMe, combined with
  `thermal='lumped'` electro-thermal coupling, `particle mechanics='swelling and cracking'`,
  `SEI on cracks` and `stress-driven LAM` — the complete
  *stress → cracking → SEI-on-cracks → capacity fade* chain for silicon anodes.
- **Traceable virtual dataset.** 31 cells (25.0–40.0 °C, 0.5 °C grid) × 53 cycles
  (3 × C/20 formation + 50 × C/2), with **real cell temperature** (a solved variable)
  exported per sample. A `coupled_diagnostics.csv` records per-cycle particle stress,
  crack length, SEI thickness and loss of active material.
- **Frequency-domain characterization.** Time-domain small-signal EIS on the DFN model
  (1000 Hz → 0.05 Hz), with the double-layer capacitance activated via
  `'surface form': 'differential'` — producing textbook Nyquist spectra
  (high-frequency intercept → mid-frequency semicircle → diffusion tail) and
  ZARC equivalent-circuit fitting (fit RMSE ≈ 1 Ω).
- **Online estimation + data-driven prediction.** `filterpy` EKF with OCV table +
  R0/R1C1 model (SOC RMSE ≈ 2.2 % under 3 mV sensor noise), and a compact LSTM
  that predicts the next 30 cycles of SOH from the first 20 (curve MAE ≈ 1.0 % SOH,
  RUL error ≈ 3.2 cycles on unseen high-temperature cells).
- **Interactive dashboard.** Streamlit app with 4 tabs: virtual experiment data &
  coupled-field diagnostics, EKF + EIS analysis, LSTM RUL, and a full-pipeline overview.

## Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  si_halfcell_dataset.py   (PyBaMM 26.x)                              │
│  DFN (P2D) + thermal + mechanics + SEI-on-cracks + stress-driven LAM │
│  31 cells × (3 formation + 50 C/2 cycles)  →  UNIBO-like CSV         │
│  + coupled_diagnostics.csv   + OCV-SOC prior table                   │
└──────────────┬───────────────────────────────────────────────────────┘
               │
     ┌─────────┼────────────────┬──────────────────────┐
     ▼         ▼                ▼                      ▼
 SOC_Train.py  kalman_soc.py  eis_simulation.py      rul_lstm.py
 LSTM          EKF online SOC  small-signal EIS      LSTM RUL
               (RMSE ≈ 2.2 %)  → Nyquist + ZARC      (MAE ≈ 1 %, ±3 cyc)
     │              │            │                      │
     └──────────────┴────────────┴──────────────────────┘
                              ▼
                     app.py  (Streamlit dashboard, 4 tabs)
```

## Repository Structure

```
├── si_halfcell_dataset.py   # One-click generator of the DFN coupled virtual dataset
├── eis_simulation.py        # Time-domain small-signal EIS → standard Nyquist + ZARC fit
├── SOC_Train.py             # LSTM SOC training (reads the unified dataset)
├── SOH_Train.py             # CNN SOH training (reads the unified dataset)
├── kalman_soc.py            # Step 2: EKF online SOC estimation (+ demo figure)
├── rul_lstm.py              # Step 3: LSTM capacity-fade / RUL prediction
├── app.py                   # Step 4: Streamlit digital-twin dashboard (streamlit run app.py)
├── data_processing/         # Dataset loaders / normalization (UNIBO-compatible)
│   ├── unibo_powertools_data.py   # Cycle/capacity tables → SOC & SOH construction
│   └── model_data_handler.py      # Scaling & model-input formatting
├── data/                    # Datasets (generated on demand, see Quick Start)
├── results/                 # Demo figures, metrics and EIS analysis tables
├── Picture/                 # Model architecture & test-result figures
└── requirements.txt
```

## Quick Start

### 0. Requirements

```bash
pip install -r requirements.txt
# Python 3.12 · PyBaMM 26.4.1 · TensorFlow 2.16 / Keras 3 · Streamlit 1.63
```

### 1. Generate the virtual dataset (required, ~10–40 min on 12 parallel workers)

```bash
python si_halfcell_dataset.py
```

Outputs to `data/si-c-half-cell/`:
`test_result.csv` (≈1.59 M rows), `test_result_trial_end.csv` (capacity/SOH, 3 100 rows),
`coupled_diagnostics.csv` (stress / crack / LAM / SEI / temperature per cycle),
`si_c_half_cell_soc_train.csv` (OCV–SOC prior table) and a capacity-fade figure.

Fast smoke test first (2 cells, 5 cycles, ~30 s):

```bash
# Windows PowerShell
$env:SI_HALFCELL_DEBUG='1'; python si_halfcell_dataset.py
```

### 2. Teach the dashboard (EKF demo figure + RUL model)

```bash
python kalman_soc.py     # EKF demo → results/digital_twin/ekf_soc_demo.png
python rul_lstm.py       # LSTM training + metrics → results/digital_twin/
```

### 3. Optional: train the SOC / SOH networks

```bash
python SOC_Train.py      # → results/trained_model/…_lstm_soc_percentage.keras
python SOH_Train.py      # → results/trained_model/…_cnn_soh_percentage.keras
```

### 4. Optional: EIS impedance analysis (~8 min, 4 T × 5 SOC × 13 frequencies)

```bash
python eis_simulation.py            # full sweep → results/eis/
# $env:EIS_QUICK='1' python eis_simulation.py    # smoke test (~1 min)
```

### 5. Launch the digital twin dashboard

```bash
streamlit run app.py
```

Tabs: **① Virtual data & coupled fields · ② EKF online SOC + EIS · ③ LSTM RUL · ④ Pipeline overview**

## Dataset Specification

| Item | Value |
|---|---|
| Chemistry | Amorphous-Si working electrode / Li-metal counter electrode (half-cell) |
| Model | DFN (P2D) + lumped thermal + swelling/cracking mechanics + SEI-on-cracks + stress-driven LAM |
| Cells | 31 (ID 000–030), ambient 25.0–40.0 °C, step 0.5 °C |
| Protocol | 3 × C/20 formation (excluded) + 50 × C/2 discharge/charge |
| Resampling | 511/512 points per half-cycle (capacity-grid, for CNN/LSTM windows) |
| Capacity fade (50 cyc) | 19.8 % @ 25 °C → 25.5 % @ 40 °C (SEI dominates, Arrhenius) |
| Auxiliary tables | SOH (capacity) table, per-cycle coupled-field diagnostics, OCV–SOC prior |

## Key Results (measured)

| Metric | Value |
|---|---|
| EKF online SOC (C/2, 3 mV noise, 15 % initial guess error) | **RMSE 2.22 %** vs 14.2 % coulomb counting |
| LSTM RUL (10 unseen high-temperature cells) | curve **MAE 1.0 % SOH**, RUL error **≈ 3.2 cycles** |
| EIS (DFN, 1000→0.05 Hz) | R0 ≈ **11.5 Ω**; Rct 70–212 Ω (↓ with T, U-shape vs SOC); ZARC fit RMSE ≈ **1 Ω** |
| EIS ↔ EKF cross-check | frequency-domain R0 ≈ 11.5 Ω vs time-domain EKF identification 17.5 Ω (same order) |

## Data & References

- **UNIBO Powertools dataset** (original loader compatibility):
  download from Mendeley `n6xg5fzsbv/1` into `data/unibo-powertools-dataset/` — see `data/README.md`.
- **PyBaMM** — *Python Battery Mathematical Modelling*, JORS 9(1):14, 2021.
- **O'Kane et al.**, *Lithium-ion battery degradation: how to model it.*
  Phys. Chem. Chem. Phys. 24:7909–7922, 2022 (OKane2022 parameter set).
- **Chen et al.**, *Development of Experimental Techniques for Parameterization of
  Multi-scale Lithium-ion Battery Models.* JES 167(8):080534, 2020 (Chen2020 parameters).
- **Wong et al.**, *Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at
  Various Battery Specifications and Discharge Cycles.* GoodIT '21 (base architecture
  of the UNIBO-compatible training pipeline).

## Acknowledgements

The repository builds on the open-source PyBaMM ecosystem and the battery-state-estimation
codebase by Kei Long Wong, chchen59, et al.; the DFN coupled-aging workflow, virtual dataset,
EIS simulation and dashboard were extended on top of it. For research and educational
use — the parameter sets and virtual data are not claimed to represent any real cell.
