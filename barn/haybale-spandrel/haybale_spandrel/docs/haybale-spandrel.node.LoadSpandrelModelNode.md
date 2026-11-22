# Load Spandrel Model

`haybale-spandrel:node:LoadSpandrelModelNode` · kind: node

Loads a spandrel (PyTorch image) model onto the GPU or CPU, for Upscale Image or Inpaint

## Ports

| id | direction | type | description |
|---|---|---|---|
| `status.loaded` | config | `haywire-core:type:STRING` | Architecture, scale, channels, device and precision of the loaded model |
| `loaded` | outlet | `haywire-core:type:ADD` | Connect it to the node that runs the model; it takes that node's kind of model |

## Settings

| name | bag | default | description |
|---|---|---|---|
| `device` | settings | `None` | Where models run: auto (CUDA, then Apple GPU, then CPU), cpu, mps, cuda or cuda:N; type another device and press Enter |
| `precision` | settings | `None` | Number format the model computes in: auto picks the fastest the model and device support; a choice the model cannot run falls back to fp32 |
| `loaded` | status | `'< shows after the first run >'` | Architecture, scale, channels, device and precision of the loaded model |

## Notes

Loads a spandrel (PyTorch image) model onto the GPU or CPU, for Upscale Image or Inpaint.

Connect the `Model` outlet first, to the node that runs the model: the outlet
takes the kind of model that node needs (an image model for Upscale Image, an
inpainting model for Inpaint, any kind for Spandrel Model Info), and the
`Model` picker appears, offering only models of that kind. The outlet keeps its
kind until the node is reset. A model of another kind, picked as a file, is
reported when it loads.

Pick the model on the card: a catalog entry, or a file from the Models folder
or the Hugging Face cache. A model that is not on this machine is an error
until it is fetched in the Models editor. The architecture is detected from
the file (`.pth`, `.safetensors`, `.ckpt`, `.pt`).

- `Device`: where the model runs. `auto` uses a CUDA GPU, else the Apple GPU,
  else the CPU. A device this machine lacks runs on the CPU, with a note.
- `Precision`: `auto` runs in half precision on a GPU when the model supports
  it, else full. A precision the model cannot run falls back to fp32, with a
  note.

- Models of the non-commercial architectures (CodeFormer, SRFormer, Restormer
  and others) load only with the library setting *Non-commercial architectures*
  on, and say so on the card: their code is licensed for non-commercial use only.

Device and precision default to the library's *Spandrel* settings. Nodes that pick the same
model with the same device and precision share one loaded copy. The model is
loaded when the graph runs and freed when it stops.
