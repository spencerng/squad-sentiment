# squad-sentiment

Experiment code and [related paper](assets/semantic-perturbations-bert-performance.pdf) to determine the effects of semantic perturbations on a BERT model trained on the [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/), where verbs in question/context pairings are modified to dissimilar verbs as adversarial examples.

Created by Spencer Ng, [Lucy Teaford](https://github.com/lucyteaford), [Andy Yang](https://github.com/oysterclam), and [Isaiah Zwick-Schachter](https://github.com/isaiahzs) for CMSC 25610: Undergraduate Computatational Lingustics at the University of Chicago (Fall 2021)

## Setup

1. Download the SQuAD 1.1 dataset:

```
mkdir squad-data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O squad-data/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O squad-data/dev-v1.1.json
```

2. Install required packages for the scripts: `pip3 install -r requirements.txt`

## Data pipeline

1. Cleanup data and generate categories for the test set: 

```
mkdir data
python3 cleanup.py
```

This will output four files:
* `data/qa.csv`: questions, expected answers, associated context IDs, and the start positions for answers for training questions
* `data/contexts.csv`: context paragraphs and related IDs that contain answers to training questions
* `data/test_questions.csv`: questions, ground truth answers (delimited by `|`), context IDs, and the category of questions in the test set (using the SQuAD dev set questions)
* `data/test_contexts.csv`: context paragraphs and related IDs that contain answers to test set questions

2. Train model:

```
mkdir models
python3 train.py
```

This will generate a baseline PyTorch-based model finetuned through Huggingface using the `ber-base-uncased` pretrained model at `models/squad-bert`.

3. Generate adversarial examples:

```
python3 synonym_replace.py
```

This will output modified contexts and/or questions using `test_questions.csv` and `test_contexts.csv` using cosine similarity levels of 0.2, 0.3, 0.4, 0.5, and 0.6 using a similarity threshold-based approach while ensuring verbs are conjugated in the same form as the original verbs in questions/pararaphs. Files are output as `data/test_contexts<similarity>.csv` or `data/test_questions<similarity>.csv`.

4. Generate predictions for a given test set:

```
mkdir results
python3 predict.py [--suffix <cosine similarity> [--modify_question] [--modify_context]]
```

If the `--modify_question` or `--modify_context` flags are respectively set, adversarial examples of questions and/or paragraph contexts of modified verbs will be evaluated at the cosine similarity level given by `--sufix`. If the flags are not set, the baseline set of `test_questions.csv` and `test_contexts.csv` will be used instead.

Results will be saved in `results/predictions[cos similartiy][_modctx][_modq].json` depending on which flags are set with question ID-predicted answer pairings in a JSON dictionary.

5. Evaluate the accuracy of the generated predictions:

```
python3 evaluate.py <prediction JSON file>
```

This will print the F1, accuracy, sample size, and number of correct answers for a given prediction JSON file, along with a breakdown for each question cateogry type.