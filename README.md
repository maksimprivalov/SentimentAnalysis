# Sentiment Analysis

## Setup

cd SentimentAnalysis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## Data

- **Amazon Reviews**: put `train.csv` and `test.csv` in `Amazon_Reviews/` (or set `--data_dir`).
- **GloVe**: download e.g. [Wikipedia 2014 + Gigaword](https://nlp.stanford.edu/projects/glove/) (e.g. `glove.6B.zip`) or the 2024 wiki+giga 100d file. Unzip and point `--glove_path` to the `.txt` file (e.g. `glove.6B.100d.txt` or your `wiki_giga_2024_100...txt`).
