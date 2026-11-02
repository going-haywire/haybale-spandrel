# Spandrel model catalog

Each `.json` file here is one model catalog entry, registered as
`haybale-spandrel:catalog:<file name without .json>`.

All four are copied verbatim from the
[OpenModelDB data repository](https://github.com/OpenModelDB/open-model-database),
`data/models/`, at commit `782aac088bd83dd3fc438a42eb0c3868dc559110`, on 2026-11-02.
The models are Real-ESRGAN releases by xinntao, licensed BSD-3-Clause:

- `4x-realesrgan-x4plus.json`: general photos, 4x (ESRGAN).
- `2x-realesrgan-x2plus.json`: general photos, 2x (ESRGAN).
- `4x-realesr-general-x4v3.json`: general photos, 4x, small and fast (Compact).
- `4x-realesr-animevideo-v3.json`: anime video frames, 4x, small and fast (Compact).

No inpainting model is listed: the published LaMa files are a TorchScript export
and a Lightning checkpoint, and spandrel loads neither as it is.
