# Spandrel

Upscale and restore images with PyTorch models (spandrel)

## Nodes
### Image
- **Inpaint** — Fills the masked part of an image with a loaded inpainting model
- **Upscale Image** — Enlarges or restores an image with a loaded model

### Models
- **Load Spandrel Model** — Loads a spandrel (PyTorch image) model onto the GPU or CPU, for Upscale Image or Inpaint
- **Spandrel Model Info** — Tells what a loaded spandrel model is: architecture, purpose, scale, device, precision

## Macros
- **Upscaler** — 

## Types
- **Image Model** — A loaded image-to-image model: upscaling and restoration
- **Inpainting Model** — A loaded model that fills a masked part of an image
- **Loaded Model** — A model loaded on a device, ready to run. Never saved with the graph.

## Settings
- **Spandrel** — Device, precision and memory for models

## Panels
- **Spandrel** — Device, precision and memory budget for spandrel models.
