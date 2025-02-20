# Project Setup and Execution Guide

This guide provides instructions for setting up your development environment and running large-scale training on Lambda Labs via SSH.

## Prerequisites

- Python 3.x
- CUDA 12.1 compatible GPU
- Lambda Labs account
- SSH client (e.g., Terminal, PuTTY)

## Setting Up Virtual Environment

1. **Create a Virtual Environment:**

   ```bash
   python3 -m venv myenv
   ```

2. **Activate the Virtual Environment:**

   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```
   - On Windows:
     ```bash
     .\myenv\Scripts\activate
     ```

3. **Install Required Packages:**

   ```bash
   # Install PyTorch and dependencies
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables:**

   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your actual tokens:
   ```
   HF_TOKEN=your_huggingface_token_here
   WANDB_API_KEY=your_wandb_key_here
   ```

## Running Training

1. **SSH to Lambda Labs Instance:**
   ```bash
   ssh <username>@<instance-ip>
   ```

2. **Clone and Setup:**
   ```bash
   git clone <your-repo>
   cd <your-repo>
   python3 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Training:**
   ```bash
   chmod +x scripts/train.sh
   ./scripts/train.sh
   ```

## Project Structure

- `configs/` - YAML configuration files
- `training/` - Training scripts
- `data/` - Dataset files
- `scripts/` - Utility scripts
- `utils/` - Helper functions

## Configuration

Training parameters can be modified in `configs/pixtral_12b_casters.yaml`