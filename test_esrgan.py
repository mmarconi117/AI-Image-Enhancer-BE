from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from PIL import Image
import numpy as np
import torch
import time

start = time.time()

print("📸 Loading image...")
img = Image.open("assets/MePro.jpg").convert("RGB")

print("🧠 Initializing model...")
model = RealESRGANer(
    scale=4,
    model_path="weights/RealESRGAN_x4plus.pth",
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32),
    tile=160,   # <--- add this
    tile_pad=10,
    pre_pad=0,
    half=False
)

print("⚙️ Starting enhancement...")
output, _ = model.enhance(np.array(img))

print(f"✅ Done in {round(time.time() - start, 2)}s. Saving output...")
Image.fromarray(output).save("output.jpg")
print("✅ Enhanced image saved as output.jpg")
