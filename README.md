# Clip-Guided Diffusion for Tiny ImageNet

This repository contains a specialized CLIP-guided diffusion pipeline optimized for the Tiny ImageNet dataset (64x64 resolution). It builds upon the DiffusionCLIP architecture to perform text-driven image manipulation, fine-tuning, and latent optimization.

## Features

- **Text-Driven Manipulation**: Edit images using simple text prompts by leveraging the power of CLIP to guide the diffusion process.
- **Multiple Operational Modes**: Support for zero-shot editing, model fine-tuning, single image editing, and latent optimization.
- **Gradio Monitoring UI**: A live web interface to monitor training losses and visualize real-time generated image transitions.
- **Configurable Pipeline**: Easily tweak diffusion steps, loss weights, dataset parameters, and text prompts via shell scripts and YAML configs.

## Installation

1. Clone this repository (or navigate to the workspace).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training & Image Generation

To start the CLIP-guided finetuning process on Tiny ImageNet, you can run the provided execution script:

```bash
bash run_script.sh
```

By default, the script takes a generic image ("a photo") and optimizes it towards the target text prompt ("a beautiful landscape"). You can modify `run_script.sh` to change these parameters, diffusion steps, and loss weights. 

Alternatively, you can run the core `main.py` directly:
```bash
python main.py --config configs/tiny_imagenet.yml \
    --clip_finetune \
    --src_txts "a photo" \
    --trg_txts "a beautiful landscape" \
    --exp "tiny_imagenet_clip_finetune" \
    # ... additional arguments
```

### 2. Gradio Monitoring UI

While the training loop is running, you can track it via the provided Gradio interface which automatically refreshes with the latest loss plots and image transitions:

```bash
python ui.py
```
Open `http://localhost:7860` in your browser to view the monitor.

## Directory Structure
- `main.py`: Entry point for configuring and invoking the diffusion pipelines.
- `diffusionclip.py`: Core model class implementing the forward/reverse diffusion loops and losses.
- `ui.py`: Gradio dashboard for real-time monitoring.
- `run_script.sh`: Main execution orchestration script.
- `configs/`: YAML configuration files (e.g., `tiny_imagenet.yml`).
- `losses/`: Implementations of directional and identity losses.

## Acknowledgements
This project is built using the principles outlined in the [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP) framework.
