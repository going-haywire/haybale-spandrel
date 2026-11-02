# Model Info

`haybale-spandrel:node:ModelInfoNode` · kind: node

Tells what a loaded model is: architecture, purpose, scale, channels, device and precision

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-spandrel:type:MODEL` | A model loaded on a device, ready to run. Never saved with the graph. |
| `architecture` | outlet | `haywire-core:type:STRING` | Text data |
| `purpose` | outlet | `haywire-core:type:STRING` | Text data |
| `scale` | outlet | `haywire-core:type:INT` | Whole number |
| `input_channels` | outlet | `haywire-core:type:INT` | Whole number |
| `output_channels` | outlet | `haywire-core:type:INT` | Whole number |
| `tiling` | outlet | `haywire-core:type:STRING` | Text data |
| `supports_fp16` | outlet | `haywire-core:type:BOOL` | True or False |
| `supports_bf16` | outlet | `haywire-core:type:BOOL` | True or False |
| `min_size` | outlet | `haywire-core:type:INT` | Whole number |
| `multiple_of` | outlet | `haywire-core:type:INT` | Whole number |
| `device` | outlet | `haywire-core:type:STRING` | Text data |
| `precision` | outlet | `haywire-core:type:STRING` | Text data |
| `parameters` | outlet | `haywire-core:type:INT` | Whole number |

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
