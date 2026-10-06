# UNIBO Powertools Dataset

The UNIBO Powertools Dataset has been collected in a laboratory test by an Italian Equipment producer. The cycling experiments are designed to analyze different cells intended for use in various cleaning equipment such as vacuum and automated floor cleaners. The vast dataset is composed of 27 batteries. The main features of the dataset are:
(1) the use of batteries from different manufacturers, (2) cells with several nominal capacities, (3) cycling is performed until the cell's end-of-life and thus data regarding the cell at different life stages are produced. Three types of tests have been conducted. (I) The standard test, where the battery was discharged at 5A current in main cycles. (II), the high current test, where the battery was discharged at 8A current in main cycles. (III), the preconditioned test, where the battery cells are stored at 45°C environments for 90 days before conducting the test. During discharge, the sampling period is 10 seconds. The experiments were conducted using the following procedure: 

1) Charge cycle: Constant Current-Constant Voltage (CC-CV) at 1.8A and 4.2V (100mA cut-off )
2) Discharge cycle: Constant Current until cut-off voltage (2.5V)
3) Repeat steps 1 and 2 (main cycle) 100 times
4) Capacity measurement: charge CC-CV 1A 4.2V (100mA cut-off ) and discharge CC 0.1A 2.5V
5)  Repeat the above steps until the end of life of the battery cell

Data are described in [data-description.md](./data-description.md).

*test_result.csv* contains all the records but the last one of each charge/discharge run. *test_result.csv* contains the last record of each run. A python file for loading data is available at https://github.com/KeiLongW/battery-state-estimation

# Paper
If you use this dataset, please cite our paper:

Kei Long Wong, Michael Bosello, Rita Tse, Carlo Falcomer, Claudio Rossi, Giovanni Pau. 2021. Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at Various Battery Specifications and Discharge Cycles. In Conference on Information Technology for Social Good (GoodIT ’21), September 9–11, 2021, Roma, Italy. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3462203.3475878