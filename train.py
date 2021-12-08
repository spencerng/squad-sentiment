import random
import pandas as pd
import datasets
import transformers

from datasets import load_dataset, load_metric, ClassLabel, Sequence, Dataset
from tqdm import tqdm
from transformers import (
    default_data_collator,
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
)


def train(data_folder="data", batch_size=4):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # The maximum length of a feature (question and context)
    max_length = 384

    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = 128

    qas = pd.read_csv(f"{data_folder}/qa.csv")
    contexts = (
        pd.read_csv(f"{data_folder}/contexts.csv").set_index("id").to_dict()["context"]
    )
    qas["context"] = [contexts[c] for c in qas["context"]]

    qas = Dataset.from_pandas(qas)

    tokenized_train_set = qas.map(
        prepare_train_features, batched=True, remove_columns=qas.column_names
    )

    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    args = TrainingArguments(
        f"bert-uncased-finetuned-squad",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_train_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("test-squad-trained")


def prepare_train_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Offset mappings give a map from token to character position in the original context
    offset_mapping = tokenized_examples.pop("offset_mapping")
    start_positions = list()
    end_positions = list()

    invalid_entries = list()

    for i, offsets in enumerate(offset_mapping):
        answer = examples["answer"][i]

        if isinstance(answer, float) or tokenized_examples is None or answer is None:
            invalid_entries.append(i)
            start_positions.append(-1)
            end_positions.append(-1)
            continue

        # Start/end character index of the answer in the text.
        start_char = examples["start_pos"][i]
        end_char = start_char + len(answer)

        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise put start/end positions
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions

    for i in invalid_entries:
        for key in tokenized_examples.keys():
            del tokenized_examples[key][i]

    return tokenized_examples


if __name__ == "__main__":
    train()
