# Solar Forecast

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
