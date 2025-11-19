# Electrical Power Prediction of Micro Gas Turbine using DL
# Problem Statement
Gas turbines play a critical role in energy production, and the dynamic relationships between input control signals and output power responses are complex. Existing physical models often cannot accurately predict the entire behavior of the system or are too complex. This project aims to improve modeling accuracy and enhance decision support processes in energy systems by predicting turbine behavior using a data-driven approach. The source of motivation is the lack of data-driven time series modeling in the literature and the need for precise predictions in the industry. 
# Dataset Description
The dataset used in this project consists of eight separate CSV files, each corresponding to a different experimental run performed on a gas turbine system. The goal of these experiments is to observe how the gas turbine responds specifically its electrical power output after applying different input voltage excitation patterns.
# Dataset Structure
Each experiment is provided as an independent, standalone CSV file, allowing the model to learn generalizable dynamic patterns from multiple conditions rather than from one long single experiment.

The dataset is divided into:

Training Experiments (6 files)

ex_1.csv

ex_9.csv

ex_20.csv

ex_21.csv

ex_23.csv

ex_24.csv

Testing Experiments (2 files)

ex_4.csv

ex_22.csv

Each file contains three readings:
1- Time:	Timestamp in seconds
2- Input voltage:	Control input signal applied to the turbine
3- Electrical power:	Electrical power output of the gas turbine
