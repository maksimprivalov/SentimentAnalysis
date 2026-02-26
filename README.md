# Sentiment Analysis

## Setup

cd SentimentAnalysis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## Results

| Model | Data | Epochs | Accuracy | F1 | Time|
|--------------------------|--------------|---|---------|------|----------|
| DistilBERT (Transformer) | 10k/class    | 3 | 89.62% | 89.62% | 3.4 min |
| DistilBERT (Transformer) | 500k/class   | 1 | 92.68% | 92.68% | 53 min  |
| CNN + GloVe              | 500k/class   | 10| 90.46% | 90.46% | 9.5 h   |

CNN + GloVe 30k/class 5 0.8643 0.8643 17.5min
CNN + GloVe 60k/class 8 0.8810 0.8809 85min


## Data

- **Amazon Reviews**: put `train.csv` and `test.csv` in `Amazon_Reviews/` (or set `--data_dir`).
- **GloVe**: download e.g. [Wikipedia 2014 + Gigaword](https://nlp.stanford.edu/projects/glove/) (e.g. `glove.6B.zip`) or the 2024 wiki+giga 100d file. Unzip and point `--glove_path` to the `.txt` file (e.g. `glove.6B.100d.txt` or your `wiki_giga_2024_100...txt`).
