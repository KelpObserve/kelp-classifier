"""Minimal Clay v1.5 smoke test: load, encode a random chip, verify GPU output."""
import time
import numpy as np
import torch
from claymodel.module import ClayMAEModule
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

CKPT = str(config.CLAY_CKPT)

print(f"torch {torch.__version__}  cuda? {torch.cuda.is_available()}  gpu: {torch.cuda.get_device_name(0)}")

print("loading Clay v1.5 checkpoint ...")
t0 = time.time()
model = ClayMAEModule.load_from_checkpoint(
    CKPT, model_size="large", shuffle=False, mask_ratio=0.0
).eval().to("cuda")
print(f"  loaded in {time.time()-t0:.1f}s")

# fake 4-band S2 chip (blue, green, red, NIR)
pixels = torch.rand(1, 4, 224, 224, device="cuda", dtype=torch.float32)
time_vec = torch.tensor([np.sin(2*np.pi*23/52), np.cos(2*np.pi*23/52), 0.0, 0.0], device="cuda", dtype=torch.float32).unsqueeze(0)
latlon = torch.tensor([np.sin(np.radians(38.4)), np.cos(np.radians(38.4)),
                       np.sin(np.radians(-9.2)), np.cos(np.radians(-9.2))],
                      device="cuda", dtype=torch.float32).unsqueeze(0)
waves = torch.tensor([0.493, 0.560, 0.665, 0.842], device="cuda", dtype=torch.float32)
gsd = torch.tensor(10.0, device="cuda", dtype=torch.float32)

datacube = {
    "pixels": pixels, "time": time_vec, "latlon": latlon,
    "gsd": gsd, "waves": waves, "platform": "sentinel-2-l2a",
}

print("running encoder ...")
t0 = time.time()
with torch.no_grad():
    out = model.model.encoder(datacube)
if isinstance(out, tuple):
    out = out[0]
print(f"  encoder output shape: {out.shape}   dtype={out.dtype}   device={out.device}")
print(f"  time: {(time.time()-t0)*1000:.0f} ms")
print(f"  GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("OK")
