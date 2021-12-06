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


# Training + dataset loading code
# Code modified from https://huggingface.co/transformers/custom_datasets.html
def read_squad(data_file="data/qa.csv", contexts="data/contexts.csv", train=True):
    contexts = pd.read_csv(contexts).set_index("id").to_dict()["context"]
    questions = pd.read_csv(data_file)

    contexts_list = list(
        map(lambda q: contexts[q[1]["context_id"]], questions.iterrows())
    )
    questions_list = list(questions["question"])

    if train:
        answers = list(
            map(
                lambda q: {"start": q[1]["start_pos"], "text": q[1]["answer"]},
                questions.iterrows(),
            )
        )
    else:
        answers = list(
            map(lambda q: str(q[1]["answer"]).split("|")[0], questions.iterrows())
        )

    return contexts_list, questions_list, answers


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def add_end_idx(answers, contexts):
    to_delete = list()
    for i, (answer, context) in enumerate(zip(answers, contexts)):
        gold_text = answer["text"]
        if isinstance(answer["text"], float):
            print(answer["text"])
            to_delete.append(i)
            continue

        start_idx = answer["start"]
        end_idx = start_idx + len(gold_text)

        answer["end"] = end_idx

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer["end"] = end_idx
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            answer["start"] = start_idx - 1
            answer["end"] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            answer["start"] = start_idx - 2
            answer["end"] = end_idx - 2  # When the gold label is off by two characters
    for i in to_delete:
        del answers[i]
        del contexts[i]


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]["start"]))
        end_positions.append(encodings.char_to_token(i, answers[i]["end"] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({"start": start_positions, "end": end_positions})


def train(train_dataset, save="models/baseline_model.ckpt"):
    # Initializing a BERT bert-base-uncased style configuration
    config = BertConfig()

    # Initializing a model from the bert-base-uncased style configuration
    model = BertForQuestionAnswering(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        torch.save(model.state_dict(), save)
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start"].to(device)
            end_positions = batch["end"].to(device)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs[0]
            loss.backward()
            optim.step()

        torch.save(model.state_dict(), save)

    model.eval()
    return model


# Train model on baseline data
def train_squad_baseline(save="models/base_model.ckpt"):
    train_contexts, train_questions, train_answers = read_squad()

    add_end_idx(train_answers, train_contexts)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_encodings = tokenizer(
        train_contexts, train_questions, truncation=True, padding=True
    )
    add_token_positions(train_encodings, train_answers)

    train_dataset = SquadDataset(train_encodings)
    model = train(train_dataset, save)
    torch.save(model.state_dict(), save)


def eval_squad(save="models/baseline_model.ckpt"):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = BertConfig()
    baseline_model = BertForQuestionAnswering(config)
    baseline_model.load_state_dict(torch.load(save))
    baseline_model.to(device)
    baseline_model.eval()

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    test_contexts, test_questions, test_answers = read_squad(
        data_file="data/testset.csv", train=False
    )
    test_encodings = tokenizer(
        test_contexts, test_questions, truncation=True, padding=True
    )
    test_encodings.update({"answer": [i for i in range(len(test_answers))]})

    test_dataset = SquadDataset(test_encodings)

    predictions = open("predictions.txt", "w+")
    predictions.write("pred,ans\n")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, sample in tqdm(enumerate(test_loader)):

        input_ids = sample["input_ids"].to(device)
        attention_mask = sample["attention_mask"].to(device)
        outputs = baseline_model(
            input_ids,
            attention_mask=attention_mask,
        )

        start = np.argmax(outputs.start_logits.cpu().detach())
        end = np.argmax(outputs.end_logits.cpu().detach())
        print(f"{start}, {end}, {sample['answer']}")
        pred_answer = test_contexts[i][start : end + 1]

        # print(pred_answer, test_contexts[i])
        predictions.write(f"{pred_answer},{test_answers[sample['answer']]}\n")

    predictions.close()


if __name__ == "__main__":
    # cleanup("./squad/train-v1.1.json", "./squad/dev-v1.1.json")
    # train_model_baseline()
    eval_squad()
