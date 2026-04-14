# FTL_SMART_BUILDING / MLP + LSTM Energy Forecasting (ASHRAE)

## 1. Run Preprocessing First
Before running any notebook, generate the cleaned datasets.

### Dataset download (Kaggle)
The raw ASHRAE dataset is hosted on Kaggle. Download the competition data, then place the CSVs in `ashrae/`:

Required files:
- `train.csv`
- `weather_train.csv`
- `building_metadata.csv`

Kaggle link:
```text
https://www.kaggle.com/competitions/ashrae-energy-prediction/data
```

From the project root:

```bash
python ashrae/preprocess_isamu_matt.py --input-dir ashrae --output-dir ashrae
```

This creates:
- `ashrae/train.cleaned_isamu_matt.csv`
- `ashrae/weather_train.cleaned.csv`
- `ashrae/train.cleaned_isamu_matt.anomaly_report.csv`

Then create the filtered training file used by experiments:

```bash
python ashrae/make_building_mean_y_ge_1.py
```

This creates:
- `ashrae/train.cleaned_isamu_matt.building_mean_y_ge_1.csv`

## 2. Notebook Order
Run notebooks in this order (all notebooks are in the project root):

1. `01_preprocessing.ipynb`
2. `02_basic_strategies.ipynb`
3. `03_mlp.ipynb`
4. `04_lstm.ipynb`

## 3. Training / Loading Modes
Both `03_mlp.ipynb` and `04_lstm.ipynb` support two modes:

- `RUN_MODE = "train"`: trains models and saves weights/results to `saved_models/`
- `RUN_MODE = "load"`: loads saved weights/results and skips training

Make sure `saved_models/` exists (it is created automatically in train mode).

## 4. Project Structure
```text
FTL_SMART_BUILDING/
  ashrae/
    preprocess_isamu_matt.py
    make_building_mean_y_ge_1.py
    train.csv
    weather_train.csv
    building_metadata.csv
    ... generated cleaned files ...
  01_preprocessing.ipynb
  02_basic_strategies.ipynb
  03_mlp.ipynb
  04_lstm.ipynb
  src/
  saved_models/
  saved_results/
```
