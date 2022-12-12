import os, torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL

# Location of current file to help locate the image 
root_location = os.path.dirname(__file__)
# This is where you specify which file you want to test
image_path_relative_to_this_file = "sample.png"

# Load stable diffusion
repo_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(repo_id, safety_checker=None)
vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae")

# Load image
image = Image.open(f"{root_location}/{image_path_relative_to_this_file}").convert("RGB")

w,h = 512, 512

# Encode image to latent
image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
image = np.array(image).astype(np.float32) / 255.0
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image)
image = 2.0 * image - 1.0
init_latent_dist = vae.encode(image).latent_dist
init_latents = init_latent_dist.sample()

# Decode latent to image
image = vae.decode(init_latents).sample
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].show()
pil_images[0].save("sample_from_vae.png")