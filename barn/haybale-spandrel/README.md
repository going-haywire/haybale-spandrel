# Spandrel

## Licences

This library's own code is MIT. With the setting *Non-commercial architectures* on (it is off by default), it also runs code from [`spandrel_extra_arches`](https://github.com/chaiNNer-org/spandrel/tree/main/libs/spandrel_extra_arches), whose architectures are licensed for **non-commercial use only**. With the setting off, that code is installed but never imported.

| Architecture | Licence |
|---|---|
| AdaCode | CC BY-NC-SA 4.0 |
| CodeFormer | S-Lab License 1.0 |
| DDColor | Alibaba, academic and non-commercial use only |
| FeMaSR | CC BY-NC-SA 4.0 |
| M3SNet | Academic Public License |
| MAT | CC BY-NC 4.0 |
| MIRNet2 | Academic Public License |
| MPRNet | Academic Public License |
| Restormer | Academic Public License |
| SRFormer | CC BY-NC 4.0 |

The package's metadata says "MIT", which covers only its wrapper; each architecture's licence is in its `__arch/LICENSE` in the spandrel repository.

Model weights carry their own licence, whichever architecture runs them: check it on the model's page.

<!-- marketstall:share-url:start -->
Always the latest (tracks the current branch):

```sh
https://github.com/going-haywire/haybale-spandrel/blob/master/marketstall.toml
```

Frozen to this version:

```sh
https://github.com/going-haywire/haybale-spandrel/blob/v0.0.4/marketstall.toml
```
<!-- marketstall:share-url:end -->

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
