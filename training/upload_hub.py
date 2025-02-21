#!/usr/bin/env python
import os
import argparse
from huggingface_hub import HfApi
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel
import torch
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
    parser.add_argument("--model-id", type=str, default="mistral-community/pixtral-12b")                   
    parser.add_argument("--output-dir", type=str, default="pixtral-12b-casters-qlora", help="Directory of the trained model")
    parser.add_argument("--repo-id", type=str, default="pixtral-12b-casters", help="Repository ID on Hugging Face Hub")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face access token")      
    return parser.parse_args()

def main():
    args = parse_args()

    # Load processor (assumes it was saved in the output directory)
    processor = AutoProcessor.from_pretrained(args.model_id)

    base_model = LlavaForConditionalGeneration.from_pretrained(args.output_dir, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.bfloat16)

    model = PeftModel.from_pretrained(base_model, args.output_dir, name="casters-v1")
    model = model.merge_and_unload()
    model._hf_peft_config_loaded = False

    # Upload the folder to the Hub
    print(f"Uploading model from {args.output_dir} to repo {args.repo_id}...")
    with tempfile.TemporaryDirectory() as tmp_path:
         processor.save_pretrained(tmp_path, push_to_hub=True, repo_id=args.repo_id, private=False, commit_message="upload processor")
    # with tempfile.TemporaryDirectory() as tmp_path:
    model.push_to_hub(repo_id=args.repo_id, commit_message="upload merged model")
    print("Upload complete.")

if __name__ == "__main__":
    main()