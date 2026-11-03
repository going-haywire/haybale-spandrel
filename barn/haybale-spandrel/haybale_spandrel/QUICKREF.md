# haybale-spandrel — component index (v0.0.1)

## node
- `haybale-spandrel:node:InpaintNode` — Inpaint — Fills the masked part of an image with a loaded inpainting model  _tags: inpaint, fill, remove, erase, mask, lama, model, spandrel_
- `haybale-spandrel:node:LoadModelNode` — Load Model — Loads a model onto the GPU or CPU, ready for Upscale Image, Inpaint and Model Info  _tags: model, load, spandrel, upscale, esrgan, pytorch, torch, gpu_
- `haybale-spandrel:node:ModelInfoNode` — Model Info — Tells what a loaded model is: architecture, purpose, scale, channels, device and precision  _tags: model, info, scale, architecture, spandrel_
- `haybale-spandrel:node:UpscaleImageNode` — Upscale Image — Enlarges or restores an image with a loaded model  _tags: upscale, super resolution, sr, enlarge, restore, esrgan, model, spandrel_

## type
- `haybale-spandrel:type:IMAGE_MODEL` — Image model — A loaded image-to-image model: upscaling and restoration
- `haybale-spandrel:type:INPAINT_MODEL` — Inpainting model — A loaded model that fills a masked part of an image
- `haybale-spandrel:type:MODEL_BASE` — Loaded model — A model loaded on a device, ready to run. Never saved with the graph.

## setting
- `haybale-spandrel:setting:SpandrelSettings` — Spandrel — Device, precision and memory for models

## panel
- `haybale-spandrel:panel:SpandrelSettingsPanel` — Spandrel — Device, precision and memory budget for spandrel models.
