"""SpandrelSettings — library-wide device, precision and memory budget for spandrel models."""

from haywire.api import BOOL, CHOICES, FLOAT, LibrarySettings, setting, settings

#: Suggested devices; any other torch device name (``cuda:3``) can be typed in.
DEVICES = ("auto", "cpu", "mps", "cuda", "cuda:0", "cuda:1")

#: Precisions a model can run in; ``auto`` picks the fastest one the model and device support.
PRECISIONS = ("auto", "fp32", "fp16", "bf16")


@settings(namespace="spandrel", label="Spandrel", description="Device, precision and memory for models")
class SpandrelSettings(LibrarySettings):
    """Where and how spandrel models run; Load Spandrel Model can change device and precision per node."""

    device = setting[CHOICES](
        "auto",
        label="Device",
        description=(
            "Where models run: auto (CUDA, then Apple GPU, then CPU), cpu, mps, cuda or cuda:N; "
            "type another device and press Enter"
        ),
        widget_config={"options": list(DEVICES), "new_value_mode": "add-unique"},
    )
    precision = setting[CHOICES](
        "auto",
        label="Precision",
        description=(
            "Number format the model computes in: auto picks the fastest the model and device "
            "support; a choice the model cannot run falls back to fp32"
        ),
        widget_config={"options": list(PRECISIONS)},
    )
    memory_budget = setting[FLOAT](
        0.0,
        label="Memory budget (GB)",
        description="Memory one model run may use, for choosing tile sizes; 0 derives it from the device",
        min=0.0,
        max=1024.0,
    )
    noncommercial_architectures = setting[BOOL](
        False,
        label="Non-commercial architectures",
        description=(
            "Also load AdaCode, CodeFormer, DDColor, FeMaSR, M3SNet, MAT, MIRNet2, MPRNet, Restormer "
            "and SRFormer models. Their code is licensed for non-commercial use only (CC BY-NC, S-Lab, "
            "Academic Public License); leave this off for commercial work. "
            "Applies from the next graph start."
        ),
    )
