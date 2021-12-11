import json
import pandas as pd

import sys


def print_accuracy(suffix="", results_folder="results", compare=False):
    with open(f"{results_folder}/predictions{suffix}.json") as pred:
        predictions = json.load(pred)

        ref_lookup = {x["id"]: x["prediction_text"] for x in predictions}

        with open(f"{results_folder}/predictions{suffix}_new.json", "w+") as pred2:
            json.dump(ref_lookup, pred2)

    with open(f"{results_folder}/references{suffix}.json") as ref:
        references = json.load(ref)

    ref_lookup = {x["id"]: x["answers"] for x in references}

    correct = 0
    missed_questions = list()

    if compare:
        inaccurate_ids = pd.read_csv(f"{results_folder}/inaccuracies.csv")["id"]

    inaccuracies = list()

    questions = pd.read_csv(f"data/testset{suffix}.csv")

    for prediction in predictions:
        expected_answers = ref_lookup[prediction["id"]]

        pred_ans = prediction["prediction_text"]
        if pred_ans in expected_answers:
            correct += 1
        else:
            if not compare or (compare and prediction["id"] not in set(inaccurate_ids)):
                question = questions[questions.id.isin([prediction["id"]])][
                    "question"
                ].to_numpy()[0]

                inaccuracies.append(
                    {
                        "prediction": pred_ans,
                        "expected_answers": expected_answers,
                        "question": question,
                        "id": prediction["id"],
                    }
                )

    pd.DataFrame(inaccuracies).to_csv(f"{results_folder}/inaccuracies{suffix}.csv")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print_accuracy()
    else:
        print_accuracy(suffix=sys.argv[1], compare=True)
