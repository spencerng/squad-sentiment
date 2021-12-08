#!/usr/bin/env python

import csv
import json
import json
import torch
import pandas as pd
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertModel,
    BertForQuestionAnswering,
    AdamW,
)

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset, load_metric


def cleanup(train_path, dev_path):
    with open(train_path) as json_file:
        train_data = json.load(json_file)["data"]
    with open(dev_path) as json_file:
        dev_data = json.load(json_file)["data"]

    with open("data/contexts.csv", "w") as csvfile:
        fieldnames = ["id", "context"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(train_data)):
            for j in range(len(train_data[i]["paragraphs"])):
                writer.writerow(
                    {
                        "id": str(i) + "x" + str(j),
                        "context": train_data[i]["paragraphs"][j]["context"],
                    }
                )

    with open("data/qa.csv", "w") as csvfile:
        fieldnames = ["question", "answer", "context_id", "start_pos"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(train_data)):
            for j in range(len(train_data[i]["paragraphs"])):
                for l in range(len(train_data[i]["paragraphs"][j]["qas"])):
                    writer.writerow(
                        {
                            "context_id": str(i) + "x" + str(j),
                            "question": train_data[i]["paragraphs"][j]["qas"][l][
                                "question"
                            ],
                            "answer": train_data[i]["paragraphs"][j]["qas"][l][
                                "answers"
                            ][0]["text"],
                            "start_pos": train_data[i]["paragraphs"][j]["qas"][l][
                                "answers"
                            ][0]["answer_start"],
                        }
                    )

    with open("data/testset.csv", "w") as csvfile:
        fieldnames = ["question", "answers", "context_id"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(train_data)):
            for j in range(len(train_data[i]["paragraphs"])):
                for l in range(len(train_data[i]["paragraphs"][j]["qas"])):
                    a = "|".join(
                        list(
                            set(
                                map(
                                    lambda x: x["text"],
                                    train_data[i]["paragraphs"][j]["qas"][l]["answers"],
                                )
                            )
                        )
                    )
                    writer.writerow(
                        {
                            "context_id": str(i) + "x" + str(j),
                            "question": train_data[i]["paragraphs"][j]["qas"][l][
                                "question"
                            ],
                            "answer": a,
                        }
                    )


if __name__ == "__main__":
    cleanup("./squad-data/train-v1.1.json", "./squad-data/dev-v1.1.json")
