# CodeAlpha Task 2 — Emotion Recognition from Speech

## Objective
Recognize human emotions such as happy, angry and sad from speech audio.

## Approach
- Speech preprocessing and silence trimming
- 40 MFCCs (Mel-Frequency Cepstral Coefficients)
- Noise, pitch-shift and time-stretch augmentation
- 1D CNN for local acoustic patterns
- Bidirectional LSTM for temporal patterns
- Accuracy, Precision, Recall, F1-score and confusion matrix

## Supported datasets
- RAVDESS
- TESS
- EMO-DB

## Example: RAVDESS
Place extracted WAV files under `data/ravdess/`, then:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --dataset ravdess --data_dir data/ravdess --epochs 35
streamlit run app.py
```

You may replace `ravdess` with `tess` or `emodb` and point `--data_dir` to that dataset.
