import json
import pandas as pd

with open("predictions.json") as pred:
    predictions = json.load(pred)

with open("references.json") as ref:
    references = json.load(ref)

ref_lookup = {x["id"]: x["answers"] for x in references}

correct = 0

for prediction in predictions:
    expected_answers = ref_lookup[prediction["id"]]

    pred_ans = prediction["prediction_text"]
    if (
        pred_ans in expected_answers
        or pred_ans.replace("the ", "") in expected_answers
        or pred_ans.replace("a ", "") in expected_answers
        or pred_ans.replace("an ", "") in expected_answers
    ):
        correct += 1
    else:
        print(pred_ans, expected_answers)

print(correct / len(predictions))
