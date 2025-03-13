import torch
import os
import json
import numpy as np
from PIL import Image
import folder_paths

class BatchLoadLatentImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.latents_dir = os.path.join(self.output_dir, "latents")
        
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "load_latent_and_image"
    CATEGORY = "latent"
    DISPLAY_NAME = "Batch Load Latent & Image"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filenames": ("STRING", {"multiline": True, "default": "latent_1\nlatent_2\nlatent_3"}),
                "load_directory": ("STRING", {"default": "latent"}),
            },
        }
    
    def load_latent_and_image(self, filenames, load_directory):
        # Determine load directory
        if load_directory.strip() == "" or load_directory.strip() == "latent":
            load_dir = self.latents_dir
        else:
            # Check if it's an absolute path
            if os.path.isabs(load_directory):
                load_dir = load_directory
            else:
                # If it's a relative path, make it relative to the output directory
                load_dir = os.path.join(self.output_dir, load_directory)
        
        if not os.path.exists(load_dir):
            raise ValueError(f"Directory does not exist: {load_dir}")
        
        # Parse filenames
        filenames = [f.strip() for f in filenames.split('\n') if f.strip()]
        
        if not filenames:
            raise ValueError("No filenames provided")
        
        # Lists to store loaded latents and images
        latent_samples = []
        images = []
        
        # Load each latent and image
        for filename in filenames:
            # Check for latent file
            latent_path = os.path.join(load_dir, f"{filename}.latent")
            if not os.path.exists(latent_path):
                print(f"Warning: Latent file not found: {latent_path}")
                continue
            
            # Load latent
            latent_data = torch.load(latent_path)
            latent_samples.append(latent_data["samples"])
            
            # Check for image file
            img_path = os.path.join(load_dir, f"{filename}.png")
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found: {img_path}")
                # Create a blank image if the image file is missing
                h, w = latent_data["samples"].shape[2] * 8, latent_data["samples"].shape[3] * 8
                blank_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
                images.append(torch.from_numpy(blank_img))
            else:
                # Load image
                pil_img = Image.open(img_path)
                img_np = np.array(pil_img).astype(np.float32) / 255.0
                if len(img_np.shape) == 2:  # Grayscale
                    img_np = np.stack([img_np, img_np, img_np], axis=2)
                elif img_np.shape[2] == 4:  # RGBA
                    img_np = img_np[:, :, :3]  # Remove alpha channel
                images.append(torch.from_numpy(img_np))
        
        if not latent_samples:
            raise ValueError("No valid latent files found")
        
        # Combine into batches
        latent_batch = torch.cat(latent_samples, dim=0)
        image_batch = torch.stack(images, dim=0)
        
        # Create the latent dictionary
        latent = {"samples": latent_batch}
        
        print(f"Loaded {len(latent_samples)} latent(s) and image(s) from {load_dir}")
        
        return (latent, image_batch) 