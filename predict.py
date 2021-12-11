import pandas as pd
import json

from datasets import load_metric, Dataset
from tqdm import tqdm
import collections
import argparse
import numpy as np

import sys

from transformers import (
    default_data_collator,
    BertTokenizerFast,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

# Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb


def prepare_validation_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    for example_index, example in enumerate(tqdm(examples)):
        # Indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            # No predictions, default case
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


def predict(
    suffix="",
    modify_question=True,
    modify_context=False,
    data_folder="data",
    results_folder="results",
):

    if modify_question:
        question_file = f"{data_folder}/test_questions{suffix}.csv"
    else:
        question_file = f"{data_folder}/test_questions.csv"

    if modify_context:
        context_file = f"{data_folder}/test_contexts{suffix}.csv"
    else:
        context_file = f"{data_folder}/test_contexts.csv"

    test_data = pd.read_csv(question_file)
    contexts = pd.read_csv(context_file).set_index("id").to_dict()["context"]

    test_data["context"] = [contexts[c] for c in test_data["context_id"]]
    test_data["answers"] = [ans.split("|") for ans in test_data["answer"]]
    test_data = Dataset.from_pandas(test_data)

    validation_features = test_data.map(
        prepare_validation_features,
        batched=True,
        remove_columns=test_data.column_names,
    )

    model = BertForQuestionAnswering.from_pretrained("models/squad-bert")

    args = TrainingArguments(
        f"bert-uncased-finetuned-squad",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model,
        args,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    raw_predictions = trainer.predict(validation_features)

    validation_features.set_format(
        type=validation_features.format["type"],
        columns=list(validation_features.features.keys()),
    )

    final_predictions = postprocess_qa_predictions(
        test_data, validation_features, raw_predictions.predictions
    )

    outfile = f"{results_folder}/predictions{suffix}"

    if modify_context:
        outfile += "_modctx"
    if modify_question:
        outfile += "_modq"

    with open(outfile + ".json" "w+") as pred_file:
        json.dump(final_predictions, pred_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", required=False)
    parser.add_argument("--modify_question", action="store_true")
    parser.add_argument("--modify_context", action="store_true")

    args = parser.parse_args()
    print("STARTING: ", args)
    tokenizer = BertTokenizerFast.from_pretrained("models/squad-bert")

    # The maximum length of a feature (question and context)
    max_length = 384

    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = 128
    batch_size = 4

    predict(
        suffix=args.suffix,
        modify_question=args.modify_question,
        modify_context=args.modify_context,
    )
