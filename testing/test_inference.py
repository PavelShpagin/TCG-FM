import json
from training.pipeline_utils import get_prompt
from transformers import AutoProcessor, LlavaForConditionalGeneration

if __name__ == "__main__":
    training_dataset = []
    with open("data/processed/casters_cards.jsonl", "r") as f:
        for line in f:
            training_dataset.append(json.loads(line))

    card_prompt = get_prompt(training_dataset[0])
    card_prompt = card_prompt[:card_prompt.find('[/INST]')+len('[/INST]')]

    print(card_prompt)

    processor = AutoProcessor.from_pretrained("pavelshpagin/pixtral-12b-casters")
    model = LlavaForConditionalGeneration.from_pretrained("pavelshpagin/pixtral-12b-casters")

    inputs = processor(text=card_prompt, images=[training_dataset[0]['img']], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=500)
    print(processor.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
