import torch
import os
import json
import numpy as np
from PIL import Image
import folder_paths
import traceback

class BatchLoadLatentImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.latents_dir = os.path.join(self.output_dir, "latents")
        try:
            if not os.path.exists(self.latents_dir):
                os.makedirs(self.latents_dir, exist_ok=True)
                print(f"Created latents directory: {self.latents_dir}")
            else:
                print(f"Latents directory exists: {self.latents_dir}")
        except Exception as e:
            print(f"Error checking/creating latents directory: {e}")
            traceback.print_exc()
        
    RETURN_TYPES = ("LATENT", "IMAGE")
    FUNCTION = "load_latent_and_image"
    CATEGORY = "latent"
    DISPLAY_NAME = "Batch Load Latent & Image"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filenames": ("STRING", {"multiline": True, "default": "latent_1\nlatent_2\nlatent_3"}),
                "load_directory": ("STRING", {"default": "latents"}),
            },
        }
    
    def load_latent_and_image(self, filenames, load_directory):
        try:
            # Determine load directory
            if load_directory.strip() == "" or load_directory.strip() == "latents":
                load_dir = self.latents_dir
                print(f"Using default latents directory: {load_dir}")
            else:
                # Check if it's an absolute path
                if os.path.isabs(load_directory):
                    load_dir = load_directory
                else:
                    # If it's a relative path, make it relative to the output directory
                    load_dir = os.path.join(self.output_dir, load_directory)
                print(f"Using custom directory: {load_dir}")
            
            if not os.path.exists(load_dir):
                print(f"Directory does not exist: {load_dir}")
                # Try to create it
                try:
                    os.makedirs(load_dir, exist_ok=True)
                    print(f"Created directory: {load_dir}")
                except Exception as e:
                    print(f"Error creating directory: {e}")
                    traceback.print_exc()
                    raise ValueError(f"Directory does not exist and could not be created: {load_dir}")
            
            # List files in the directory for debugging
            print(f"Files in {load_dir}:")
            for file in os.listdir(load_dir):
                print(f"  - {file}")
            
            # Parse filenames
            filenames = [f.strip() for f in filenames.split('\n') if f.strip()]
            print(f"Filenames to load: {filenames}")
            
            if not filenames:
                raise ValueError("No filenames provided")
            
            # Lists to store loaded latents and images
            latent_samples = []
            images = []
            
            # Load each latent and image
            for filename in filenames:
                try:
                    # Check for latent file
                    latent_path = os.path.join(load_dir, f"{filename}.latent")
                    print(f"Looking for latent file: {latent_path}")
                    
                    if not os.path.exists(latent_path):
                        print(f"Warning: Latent file not found: {latent_path}")
                        continue
                    
                    # Load latent
                    try:
                        latent_data = torch.load(latent_path)
                        print(f"Loaded latent from: {latent_path}")
                        print(f"Latent shape: {latent_data['samples'].shape}")
                        latent_samples.append(latent_data["samples"])
                    except Exception as e:
                        print(f"Error loading latent from {latent_path}: {e}")
                        traceback.print_exc()
                        continue
                    
                    # Check for image file
                    img_path = os.path.join(load_dir, f"{filename}.png")
                    print(f"Looking for image file: {img_path}")
                    
                    if not os.path.exists(img_path):
                        print(f"Warning: Image file not found: {img_path}")
                        # Create a blank image if the image file is missing
                        h, w = latent_data["samples"].shape[2] * 8, latent_data["samples"].shape[3] * 8
                        blank_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
                        images.append(torch.from_numpy(blank_img))
                        print(f"Created blank image of size {h}x{w}")
                    else:
                        # Load image
                        try:
                            pil_img = Image.open(img_path)
                            img_np = np.array(pil_img).astype(np.float32) / 255.0
                            print(f"Loaded image from: {img_path}, shape: {img_np.shape}")
                            
                            if len(img_np.shape) == 2:  # Grayscale
                                img_np = np.stack([img_np, img_np, img_np], axis=2)
                                print(f"Converted grayscale to RGB, new shape: {img_np.shape}")
                            elif img_np.shape[2] == 4:  # RGBA
                                img_np = img_np[:, :, :3]  # Remove alpha channel
                                print(f"Removed alpha channel, new shape: {img_np.shape}")
                                
                            images.append(torch.from_numpy(img_np))
                        except Exception as e:
                            print(f"Error loading image from {img_path}: {e}")
                            traceback.print_exc()
                            
                            # Create a blank image as fallback
                            h, w = latent_data["samples"].shape[2] * 8, latent_data["samples"].shape[3] * 8
                            blank_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
                            images.append(torch.from_numpy(blank_img))
                            print(f"Created blank image as fallback, size {h}x{w}")
                except Exception as e:
                    print(f"Error processing filename {filename}: {e}")
                    traceback.print_exc()
            
            if not latent_samples:
                print("No valid latent files found!")
                raise ValueError("No valid latent files found")
            
            # Combine into batches
            latent_batch = torch.cat(latent_samples, dim=0)
            image_batch = torch.stack(images, dim=0)
            
            print(f"Final latent batch shape: {latent_batch.shape}")
            print(f"Final image batch shape: {image_batch.shape}")
            
            # Create the latent dictionary
            latent = {"samples": latent_batch}
            
            print(f"Successfully loaded {len(latent_samples)} latent(s) and image(s) from {load_dir}")
            
            return (latent, image_batch)
            
        except Exception as e:
            print(f"Error in load_latent_and_image: {e}")
            traceback.print_exc()
            raise 