""" Adapted from the official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import sys
import os

import pandas as pd


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, original_test="data/test_questions.csv", suffix=""):
    f1 = exact_match = total = missed = 0

    questions = pd.read_csv(original_test)
    inaccuracies = list()

    category_breakdown = dict()

    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                category = questions[questions.id.isin([qa["id"]])][
                    "category"
                ].to_numpy()[0]

                if category not in category_breakdown:
                    category_breakdown[category] = {
                        "n": 0,
                        "exact_match": 0,
                        "f1": 0,
                        "missed": 0,
                    }

                total += 1
                category_breakdown[category]["n"] += 1
                if qa["id"] not in predictions:
                    missed += 1
                    category_breakdown[category]["missed"] += 1
                    continue

                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]

                correct = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths
                )

                if not correct:
                    question = questions[questions.id.isin([qa["id"]])][
                        "question"
                    ].to_numpy()[0]

                    inaccuracies.append(
                        {
                            "prediction": prediction,
                            "expected_answers": ground_truths,
                            "question": question,
                            "id": qa["id"],
                        }
                    )

                exact_match += correct
                category_breakdown[category]["exact_match"] += correct

                cur_f1 = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths
                )
                f1 += cur_f1
                category_breakdown[category]["f1"] += cur_f1

    pd.DataFrame(inaccuracies).to_csv(f"results/inaccuracies{suffix}.csv")

    acc = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    for category in category_breakdown.keys():
        category_breakdown[category]["f1"] = (
            100.0
            * category_breakdown[category]["f1"]
            / category_breakdown[category]["n"]
        )
        category_breakdown[category]["acc"] = (
            100.0
            * category_breakdown[category]["exact_match"]
            / category_breakdown[category]["n"]
        )

    return {
        "exact_match": exact_match,
        "f1": f1,
        "acc": acc,
        "n": total,
        "missed": missed,
    }, category_breakdown


def load_data(
    prediction_path, result_folder="results", test_set="squad-data/dev-v1.1.json"
):
    with open(test_set) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json["data"]
    with open(prediction_path) as prediction_file:
        predictions = json.load(prediction_file)

    suffix = prediction_path.split("/predictions")[-1].split(".json")[0]

    return evaluate(dataset, predictions, suffix=suffix)


def conv_category_stats(category_breakdown, prefix=""):
    results = list()
    for cat, stats in category_breakdown.items():
        stats["title"] = prefix
        stats["category"] = cat
        results.append(stats)
    return results


def eval_all():
    baseline, cat_breakdown = load_data("results/predictions.json")
    baseline["title"] = "baseline"
    results = [baseline]
    cat_results = conv_category_stats(cat_breakdown, "baseline")

    for suffix in ["_modq", "_modctx", "_modctx_modq"]:
        for level in [0.2, 0.3, 0.4, 0.5, 0.6]:
            if os.path.exists(f"results/predictions{level}{suffix}.json"):
                result, cat_breakdown = load_data(
                    f"results/predictions{level}{suffix}.json"
                )
                result["title"] = f"predictions{level}{suffix}"
                results.append(result)

                cat_results += conv_category_stats(
                    cat_breakdown, f"predictions{level}{suffix}"
                )

    pd.DataFrame(results).to_csv("results/all_results.csv")
    pd.DataFrame(cat_results).to_csv("results/all_cat_results.csv")


if __name__ == "__main__":
    # print(load_data(sys.argv[1]))
    eval_all()
