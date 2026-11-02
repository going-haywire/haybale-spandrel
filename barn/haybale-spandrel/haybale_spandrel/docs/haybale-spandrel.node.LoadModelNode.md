# Load Model

`haybale-spandrel:node:LoadModelNode` · kind: node

Loads a model onto the GPU or CPU, ready for Upscale Image, Inpaint and Model Info

## Ports

| id | direction | type | description |
|---|---|---|---|
| `model` | inlet | `haybale-models:type:MODEL_REF` | A model, named by a catalog entry or by a file (url and sha256). Resolve Model turns it into a path on disk. |
| `status.loaded` | config | `haywire-core:type:STRING` | Architecture, scale, channels, device and precision of the loaded model |
| `loaded` | outlet | `haybale-spandrel:type:MODEL` | A model loaded on a device, ready to run. Never saved with the graph. |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `device` | settings | `None` | Where models run: auto (CUDA, then Apple GPU, then CPU), cpu, mps, cuda or cuda:N; type another device and press Enter |
| `precision` | settings | `None` | Number format the model computes in: auto picks the fastest the model and device support; a choice the model cannot run falls back to fp32 |
| `loaded` | status | `''` | Architecture, scale, channels, device and precision of the loaded model |

## Notes

Loads a model onto the GPU or CPU, ready for Upscale Image, Inpaint and Model Info.

Pick the model on the card: a catalog entry, or a file from the Models folder
or the Hugging Face cache. A model that is not on this machine is an error
until it is fetched in the Models editor. The architecture is detected from
the file (`.pth`, `.safetensors`, `.ckpt`, `.pt`).

- `Device`: where the model runs. `auto` uses a CUDA GPU, else the Apple GPU,
  else the CPU. A device this machine lacks runs on the CPU, with a note.
- `Precision`: `auto` runs in half precision on a GPU when the model supports
  it, else full. A precision the model cannot run falls back to fp32, with a
  note.

Both default to the library's *Spandrel* settings. Load Model nodes that pick
the same model with the same device and precision share one loaded copy. The
model is loaded when the graph runs and freed when it stops.
