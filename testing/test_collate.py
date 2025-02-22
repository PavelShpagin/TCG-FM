import unittest
from transformers import AutoProcessor
import torch
from training.pipeline_utils import collate_fn
from datasets import load_dataset
import json

class TestCollate(unittest.TestCase):
    def setUp(self):
        # Initialize the processor
        self.processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
    def test_collate_and_decode(self):
        # Load a small subset of data
        training_dataset = []
        with open("data/processed/casters_cards.jsonl", "r") as f:
            for line in f:
                training_dataset.append(json.loads(line.strip()))

        batch = training_dataset[:5]
        # Process the batch using collate_fn
        processed_batch = collate_fn(batch, self.processor)
        
        # Check the structure of processed batch
        self.assertIn("pixel_values", processed_batch)
        self.assertIn("input_ids", processed_batch)
        self.assertIn("attention_mask", processed_batch)
        
        # Verify shapes
        batch_size = len(batch)
        self.assertEqual(processed_batch["pixel_values"].shape[0], batch_size)
        self.assertEqual(processed_batch["input_ids"].shape[0], batch_size)
        self.assertEqual(processed_batch["attention_mask"].shape[0], batch_size)
        
        # Try to decode the input_ids back to text
        decoded_texts = self.processor.batch_decode(processed_batch["input_ids"])
        
        print("\nDecoded texts:")
        for text in decoded_texts:
            print(text)

if __name__ == "__main__":
    unittest.main()
