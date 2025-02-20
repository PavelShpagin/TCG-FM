#!/usr/bin/env python
import os
import argparse
from huggingface_hub import HFApi
from transformers import AutoProcessor
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
    parser.add_argument("--output-dir", type=str, default="pixtral-12b-casters", help="Directory of the trained model")
    parser.add_argument("--repo-id", type=str, default="pixtral-12b-casters", help="Repository ID on Hugging Face Hub")
    parser.add_argument("--hf-token", type=str, required=True, help="Hugging Face access token")
    return parser.parse_args()

def main():
    args = parse_args()
    api = HFApi(token=args.hf_token)
    
    # Load processor (assumes it was saved in the output directory)
    processor = AutoProcessor.from_pretrained(args.output_dir)
    
    # Upload the folder to the Hub
    print(f"Uploading model from {args.output_dir} to repo {args.repo_id}...")
    api.upload_folder(folder_path=args.output_dir, repo_id=args.repo_id, commit_message="Upload from Lambda Labs training run")
    print("Upload complete.")

if __name__ == "__main__":
    main()
