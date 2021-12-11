# squad-sentiment

## Setup

## Data pipeline

1. Download SQuAD dataset:

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O squad-data/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O squad-data/dev-v1.1.json
```

2. Cleanup data and generate categories for the test set: 

```
mkdir data
python3 cleanup.py
```

3. Train model:

```
mkdir models
python3 train.py
```

4. Generate predictions for a given test set

```
mkdir results
python3 predict.py [--prefix <cosine similarity>] [--modify_question] [--modify_context]
```

5. Generate adverserial examples

```
python3 synonym_replace.py
```

6. Evaluate the accuracy of the 

```
python3 evaluate.py <prediction JSON file>
```