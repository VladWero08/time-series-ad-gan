# time-series-ad-gan
This repository contains the code and the documentation for my Dissertation thesis at the University of Bucharest, for the Artificial Intelligence master's, written during the *2025-2026* academic year. The thesis explores different prediction-based methods for time-series anomaly detection for both univariate and multivariate time-series, focusing on Generative Adversarial Networks (GANs).

## **Methodology**
In the beginning, we presented the different types of anomalies that can appear in a time series:

- **point anomalies**: a single poiny anomaly that appears *outside* the normal range of values;
- **contextual anomalies**: a single anomaly point that appears *inside* the normal range of values;
- **collective anomalies**: a subsequence of *consecutive anomalous points*, no matter if they are point or contextual anomalies. 

Afterwards, we used a taxonomy that classifies anomaly detection in time series into three categories: distance-based, density-based, and prediction-based methods. From these three categories, we were interested in analysing the prediction-based ones, which were also classified into two subcategories: forecast-based (e.g., ARIMA, LSTM), and reconstruction-based (e.g., AE, GAN). We analysed existing literature that used prediction-based methods, focusing on GAN-based methods:

- **MAD-GAN**: standard GAN;
- **TadGAN**: cycle GAN;

which were compared to three baselines: ARIMA, LSTM, and LSTM-AE.

## **Experiments**
All the experiments can be found under the *experiments/* folder, which contains all the notebooks used to evaluate LSTM, LSTM-AE, MAD-GAN, and TadGAN on the NAB, Yahoo S5, and NASA datasets. The results are in the output of the cells.

To reproduce the results, or run different experiments, in the *src/models* folder, there is a file for each model used that is structured like this:
- *pytorch.nn.Module*: the class which implementes the *forward* function 
- *train function*: full training pipeline for the model
- *test function*: full testing pipeline for the model, for MAD-GAN and TadGAN expects to receive the reconstruction errors to use; they can be multiple
- *run_pipeline function*: full training and testing pipeline, which we recommend using; it has the default parameters set, it can split the given time-series into train, test, and validation, or directly use train and test data; for each model, flexible parameters can be set up for different experiments.

For general utility functions, we have the *src/utils* folder:
- *data.py*: preprocessing, train-test-validation splitting
- *signals*: files with *torch.utils.data.Dataset* classes to split the original signals into subsequences for forecasting and reconstruction methods
- *errors*: different reconstruction errors
- *detection*: point, contextual and collective anomaly detection
- *evaluation*: computation of F1, recall and precision for point and collective anomalies

## **Results**
The results reported below correspond to the best F1-score obtained by each method on the NASA (MSL, SMAP), NAB (Art, AdEx, AWS, Traffic, Twitter), and Yahoo S5 (A1–A4) benchmark datasets. Overall, the forecasting-based methods achieved the strongest performance, with LSTM obtaining the highest average score across all datasets. Among the GAN-based approaches, TadGAN consistently outperformed MAD-GAN, suggesting that the cycle-consistency mechanism provides more robust reconstructions for anomaly detection.

| Method   | MSL  | SMAP | A1    | A2    | A3    | A4    | Art  | AdEx | AWS  | Traffic | Twitter | Mean |
|----------|------|------|-------|-------|-------|-------|------|------|------|---------|---------|------|
| LSTM     | 0.330 | 0.417 | 0.462 | **1.000** | 0.980 | **1.000** | **0.594** | **0.833** | 0.667 | **0.571** | 0.207 | **0.641** |
| ARIMA*   | **0.492** | **0.420** | **0.726** | 0.836 | 0.815 | 0.703 | 0.353 | 0.583 | 0.518 | 0.571 | **0.567** | 0.598 |
| TadGAN   | 0.366 | 0.406 | 0.537 | 0.684 | 0.160 | 0.272 | 0.389 | 0.333 | **0.806** | 0.000 | 0.194 | 0.377 |
| LSTM-AE  | 0.346 | 0.220 | 0.657 | 0.514 | 0.370 | 0.200 | 0.344 | 0.500 | 0.373 | 0.381 | 0.190 | 0.372 |
| MAD-GAN  | 0.347 | 0.345 | 0.537 | 0.357 | 0.152 | 0.090 | 0.261 | 0.333 | 0.334 | 0.071 | 0.164 | 0.271 |
