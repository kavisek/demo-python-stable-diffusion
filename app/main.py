from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, use_safetensors=True
)
prompt = "A picture of naruto uzumaki in sage mode"

image = pipeline(prompt, num_inference_steps=25).images[0]

# Save the image to disk
image.save("output.png")