# squad-sentiment

## Setup

## Data pipeline

1. Download SQuAD dataset:

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O squad-data/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O squad-data/dev-v1.1.json
```

2. Cleanup data: 

```
mkdir data
python3 cleanup.py
```

3. Train model: `python3 train.py`
4. Evaluate model
