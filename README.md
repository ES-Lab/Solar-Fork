# Solar Forecast

This is a work in progress ... The repo includes sequential learning models for solar PV forecast; model tuning,... we keep updating the repo
### Repo Structure

```
solar_fork/
│
├── models/
│   ├── forecast.py
│   │   ├── Class: ForecastModel
│   │   │   ├── Function: __init__()
│   │   │   ├── Function: build_model()
│   │   │   ├── Function: train()
│   │   │   ├── Function: predict()
│   │   │   ├── Function: save_model()
│   │   │   ├── Function: load_model()
│   │
│   ├── nn.py
│       ├── Function: build_rnn_model()
│       ├── Function: build_gru_model()
│       ├── Function: build_lstm_model()
│       ├── Function: build_cnn_model()
│       ├── Function: build_tcn_model()
│       ├── Function: build_transformer_model()
│       ├── Function: build_ann_model()
│
├── utils/
│   ├── assets.py
│   │   ├── Function: evaluate_and_plot()
│
└── data/
    ├── dataset.py
        ├── Class: SeqDataset
        │   ├── Function: __init__()
        │   ├── Function: normalize_df()
        │   ├── Function: create_sequences()
        │   ├── Function: split_data()
    ├── data_UT_clean.csv
```

## Citation
```python
[1] E. Shirazi and A. H. Ardakani, "Enhancing Convolutional Neural Network Performance for Forecasting in Energy Systems via Hyperparameter Tuning," 2025 IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT Europe), Valletta, Malta, 2025, pp. 1-5, doi: 10.1109/ISGTEurope64741.2025.11305680.
```
