import torch
import os
import json
import time
import numpy as np
from PIL import Image
import folder_paths
import traceback

class BatchSaveLatentImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.latents_dir = os.path.join(self.output_dir, "latents")
        try:
            os.makedirs(self.latents_dir, exist_ok=True)
            print(f"Latents directory created/verified: {self.latents_dir}")
        except Exception as e:
            print(f"Error creating latents directory: {e}")
            traceback.print_exc()
        
    RETURN_TYPES = ()
    FUNCTION = "save_latent_and_image"
    CATEGORY = "latent"
    DISPLAY_NAME = "Batch Save Latent & Image"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "filenames": ("STRING", {"multiline": True, "default": "latent_1\nlatent_2\nlatent_3"}),
                "save_directory": ("STRING", {"default": "latents"}),
            },
        }
    
    def save_latent_and_image(self, latent, image, filenames, save_directory):
        try:
            # Determine save directory
            if save_directory.strip() == "" or save_directory.strip() == "latents":
                save_dir = self.latents_dir
                print(f"Using default latents directory: {save_dir}")
            else:
                # Check if it's an absolute path
                if os.path.isabs(save_directory):
                    save_dir = save_directory
                else:
                    # If it's a relative path, make it relative to the output directory
                    save_dir = os.path.join(self.output_dir, save_directory)
                
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Custom directory created/verified: {save_dir}")
                except Exception as e:
                    print(f"Error creating custom directory: {e}")
                    traceback.print_exc()
                    # Fallback to default directory
                    save_dir = self.latents_dir
                    print(f"Falling back to default directory: {save_dir}")
            
            # Parse filenames
            filenames = [f.strip() for f in filenames.split('\n') if f.strip()]
            
            # Get batch sizes
            latent_batch_size = latent["samples"].shape[0]
            image_batch_size = image.shape[0]
            
            print(f"Latent batch size: {latent_batch_size}, Image batch size: {image_batch_size}")
            
            # Determine if we're in batch mode
            is_batch = latent_batch_size > 1 or image_batch_size > 1
            
            # If no filenames provided, generate default ones
            if not filenames:
                timestamp = int(time.time())
                filenames = [f"latent_{timestamp}_{i}" for i in range(max(latent_batch_size, image_batch_size))]
                print(f"Generated default filenames: {filenames}")
            
            # If not in batch mode but multiple filenames, use only the first one
            if not is_batch and len(filenames) > 1:
                filenames = [filenames[0]]
                print(f"Not in batch mode, using only first filename: {filenames[0]}")
            
            # If in batch mode but only one filename, generate more
            if is_batch and len(filenames) == 1:
                base_name = filenames[0]
                filenames = [f"{base_name}_{i}" for i in range(max(latent_batch_size, image_batch_size))]
                print(f"In batch mode with single filename, generated: {filenames}")
            
            # Ensure we have enough filenames
            while len(filenames) < max(latent_batch_size, image_batch_size):
                filenames.append(f"{filenames[-1]}_extra_{len(filenames)}")
            
            # Save each latent and image
            for i in range(max(latent_batch_size, image_batch_size)):
                try:
                    # Get the latent sample (handle batch size differences)
                    latent_idx = min(i, latent_batch_size - 1)
                    latent_sample = {
                        "samples": latent["samples"][latent_idx:latent_idx+1],
                    }
                    if "noise_mask" in latent:
                        if latent["noise_mask"] is not None:
                            latent_sample["noise_mask"] = latent["noise_mask"][latent_idx:latent_idx+1] if latent["noise_mask"].shape[0] > 1 else latent["noise_mask"]
                    
                    # Save latent
                    latent_path = os.path.join(save_dir, f"{filenames[i]}.latent")
                    try:
                        torch.save(latent_sample, latent_path)
                        print(f"Saved latent to: {latent_path}")
                    except Exception as e:
                        print(f"Error saving latent to {latent_path}: {e}")
                        traceback.print_exc()
                    
                    # Get the image (handle batch size differences)
                    image_idx = min(i, image_batch_size - 1)
                    img = image[image_idx]
                    
                    # Save image
                    img_path = os.path.join(save_dir, f"{filenames[i]}.png")
                    try:
                        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(img_path)
                        print(f"Saved image to: {img_path}")
                    except Exception as e:
                        print(f"Error saving image to {img_path}: {e}")
                        traceback.print_exc()
                    
                    # Save metadata to link latent and image
                    metadata = {
                        "latent_file": f"{filenames[i]}.latent",
                        "image_file": f"{filenames[i]}.png",
                    }
                    metadata_path = os.path.join(save_dir, f"{filenames[i]}.json")
                    try:
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)
                        print(f"Saved metadata to: {metadata_path}")
                    except Exception as e:
                        print(f"Error saving metadata to {metadata_path}: {e}")
                        traceback.print_exc()
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    traceback.print_exc()
            
            print(f"Saved {max(latent_batch_size, image_batch_size)} latent(s) and image(s) to {save_dir}")
            
        except Exception as e:
            print(f"Error in save_latent_and_image: {e}")
            traceback.print_exc()
        
        # No return values
        return () 