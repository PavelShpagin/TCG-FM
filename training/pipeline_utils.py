import json
import random
import torch

def randomly_empty_fields(input_dict, empty_prob=0.5):
    return {key: "" if random.random() > empty_prob else value for key, value in input_dict.items()}

def get_prompt(input_dict, autofill_prompt, training_dataset, examples=2):
    input_copy = input_dict.copy()
    input_copy.pop('img', None)
    examples_copy = []
    for example in training_dataset[:examples]:
        example_copy = example.copy()
        example_copy.pop('img', None)
        examples_copy.append(example_copy)
    examples_json = "\n\n".join([json.dumps(example, indent=2) for example in examples_copy])
    autofill_prompt_formatted = autofill_prompt.format(examples=examples_json)
    return (
        f"<s>[INST]{autofill_prompt_formatted}\nInput:\n[IMG]\n"
        f"{json.dumps(randomly_empty_fields(input_copy), indent=2)}\n\nOutput:[/INST]\n"
        f"{json.dumps(input_copy, indent=2)}</s>"
    )

def collate_fn(examples, processor):
    texts = [get_prompt(example, autofill_prompt="", training_dataset=examples) for example in examples]
    images = [[example.get("img", "")] for example in examples]
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    seq_len = labels.shape[1]
    seq_mask = torch.arange(seq_len)
    prompt_token_id = processor.tokenizer.convert_tokens_to_ids("[/INST]")
    pad_token_id = processor.tokenizer.pad_token_id
    prompt_mask = seq_mask[None, :] <= torch.argmax((labels == prompt_token_id).int(), dim=1)[:, None]
    padding_mask = seq_mask[None, :] > torch.argmax((labels == pad_token_id).int(), dim=1)[:, None]
    labels[prompt_mask] = -100
    labels[padding_mask] = -100
    batch["labels"] = labels
    return batch
