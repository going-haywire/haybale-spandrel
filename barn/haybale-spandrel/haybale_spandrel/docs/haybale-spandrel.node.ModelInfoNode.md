# Model Info

`haybale-spandrel:node:ModelInfoNode` · kind: node

Tells what a loaded model is: architecture, purpose, scale, channels, device and precision

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-spandrel:type:MODEL` | Loaded model to describe |
| `architecture` | outlet | `haywire-core:type:STRING` | Architecture name, such as ESRGAN |
| `purpose` | outlet | `haywire-core:type:STRING` | What the model does, such as SR or Inpainting |
| `scale` | outlet | `haywire-core:type:INT` | Upscale factor; 1 for restoration |
| `input_channels` | outlet | `haywire-core:type:INT` | Channels the model takes: 1, 3 or 4 |
| `output_channels` | outlet | `haywire-core:type:INT` | Channels the model returns |
| `tiling` | outlet | `haywire-core:type:STRING` | Whether it can run in tiles: SUPPORTED, DISCOURAGED or INTERNAL |
| `supports_fp16` | outlet | `haywire-core:type:BOOL` | Whether it runs in half precision (fp16) |
| `supports_bf16` | outlet | `haywire-core:type:BOOL` | Whether it runs in bfloat16 |
| `min_size` | outlet | `haywire-core:type:INT` | Smallest input side in pixels |
| `multiple_of` | outlet | `haywire-core:type:INT` | Input sides must be a multiple of this |
| `device` | outlet | `haywire-core:type:STRING` | Where it runs, such as mps or cuda:0 |
| `precision` | outlet | `haywire-core:type:STRING` | Number format it runs in, such as fp16 |
| `parameters` | outlet | `haywire-core:type:INT` | Number of weights |

## Notes

Tells what a loaded model is.

- `Architecture`: the network design, e.g. `ESRGAN` or `RealESRGAN Compact`.
- `Purpose`: `SR` (enlarges), `Restoration` (same size), `FaceSR` or `Inpainting`.
- `Scale`: how many times larger the result is.
- `Input Channels`, `Output Channels`: 1 (gray), 3 (RGB) or 4 (RGBA).
- `Tiling`: `SUPPORTED`, `DISCOURAGED` (runs whole unless a tile size is set) or
  `INTERNAL` (the model tiles itself).
- `fp16`, `bf16`: whether the model can run in that precision.
- `Min Size`, `Multiple Of`: input sizes the model needs; smaller or odd sizes
  are padded for it.
- `Device`, `Precision`: where and how this copy runs.
- `Parameters`: the number of weights.
