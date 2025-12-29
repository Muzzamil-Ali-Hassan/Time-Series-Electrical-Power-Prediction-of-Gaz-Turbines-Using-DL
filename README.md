# Electrical Power Prediction of Micro Gas Turbine using DL
# Project Overview
Micro gas turbines play a critical role in modern energy systems due to their compact size, high efficiency, and suitability for distributed power generation. However, accurately modeling their dynamic behavior remains a challenging task. The relationship between control inputs (such as input voltage) and the resulting electrical power output is highly nonlinear, time-dependent, and influenced by internal physical dynamics.
Traditional physics-based and thermodynamic models often require detailed system parameters and assumptions, making them complex, computationally expensive, and sometimes insufficient for real-time prediction. In contrast, data-driven deep learning models, particularly recurrent neural networks, offer a powerful alternative for learning complex temporal dependencies directly from experimental data.
This project investigates the use of Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Simple Recurrent Neural Network (RNN) architectures to predict the electrical power output of a micro gas turbine based solely on historical input voltage measurements.
# Objectives of this Project
The main objectives of this project are:
  To develop a data-driven time-series prediction framework for micro gas turbine power output.
  To preprocess and validate experimental turbine datasets recorded under different excitation conditions.
  To compare the performance of LSTM, GRU, and RNN architectures in modeling turbine dynamics.
  To evaluate model generalization using independent test experiments.
  To analyze prediction accuracy using RMSE and MAE metrics.
  To provide insights into the suitability of deep learning models for industrial energy systems.
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
# Data Validation and Preprocessing
Before model training, the preprocessing steps applied for the datasets are:
  1- Verification of missing values
  2- Detection of duplicated timestamps
  3- Sorting by time to preserve temporal order
  4- Validation of sampling interval consistency (~1 second)
  5- Feature scaling using MinMaxScaler
  6- Sliding-window sequence generation for supervised learning
  7- A 30-second historical window is used to predict the next-step electrical power output.
# Modeling Approach
Input–Output Formulation
  Input: Sequence of past input voltage values
  Output: Electrical power at the next time step
Models Implemented
Three recurrent architectures were implemented using Keras (TensorFlow backend):
  LSTM (Long Short-Term Memory)
  GRU (Gated Recurrent Unit)
  Simple RNN
Each model consists of:
  One recurrent layer with 64 hidden units
  A fully connected output layer
  Adam optimizer
  Mean Squared Error (MSE) loss function
  Early stopping to prevent overfitting
# Technologies Used
Python:
  NumPy, Pandas
  Matplotlib, Seaborn
  Scikit-learn
  TensorFlow / Keras
  Jupyter Notebook
# Results and Discussion
1- Training Dynamics and Convergence:
  LSTM showed the most stable and smooth convergence, with closely aligned training and validation loss curves. This indicates strong generalization and effective learning of long-term dependencies.
  GRU converged faster than LSTM, achieving low loss within fewer epochs. Its simpler gating mechanism offers computational efficiency while maintaining strong performance.
  RNN exhibited slower convergence and mild instability, reflecting its limited ability to handle long-term dependencies and susceptibility to vanishing gradients.
2- Predictive Performance on Test Data:
| Model| RMSE (Watts) | MAE (Watts) |
| LSTM | 346.207      | 201.098 |
| GRU  | 355.758      | 199.502 |
| RNN  | 352.778      | 213.305 |
  LSTM achieved the lowest RMSE, indicating superior handling of large prediction errors.
  GRU achieved the lowest MAE, suggesting slightly better average error performance.
  RNN showed the weakest performance overall.
# Conclusion
This project demonstrated the effectiveness of deep learning-based time-series models for predicting the electrical power output of a micro gas turbine. Among the evaluated architectures, LSTM achieved the best overall performance, while GRU offered a competitive and efficient alternative. The findings confirm that data-driven recurrent models can serve as a powerful complement to traditional physical models, enabling accurate, scalable, and adaptable turbine performance prediction.
