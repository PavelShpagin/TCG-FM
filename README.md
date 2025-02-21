# Project Setup and Execution Guide

This guide provides instructions for setting up your development environment and running large-scale training on Lambda Labs via SSH.

## Prerequisites

- Python 3.x
- CUDA 12.1 compatible GPU

## Setting Up Virtual Environment

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/PavelShpagin/TCG-FM.git
   cd TCG-FM
   ```

2. **Create a Virtual Environment:**

   ```bash
   python3 -m venv myenv
   ```

3. **Activate the Virtual Environment:**

   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```
   - On Windows:
     ```bash
     .\myenv\Scripts\activate
     ```

4. **Run Setup Script:**

   ```bash
   chmod +x scripts/setup.sh
   bash scripts/setup.sh
   ```

5. **Install HuggingFace CLI and login:**
   ```bash
   sudo apt-get install huggingface-cli
   huggingface-cli login
   ```

6. **Run Training:**
   ```bash
   chmod +x scripts/train.sh -s
   bash scripts/train.sh
   ```

7. **Upload to Hub:**
   ```bash
   chmod +x scripts/upload_hub.sh
   ./scripts/upload_hub.sh --hf_token <your_huggingface_token_here>
   ```

## Configuration

Training parameters can be modified in `configs/pixtral_12b_casters.yaml`