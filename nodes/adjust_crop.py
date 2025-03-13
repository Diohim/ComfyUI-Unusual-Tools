import torch
import numpy as np

class AdjustCrop:
    def __init__(self):
        pass

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "image/transform"
    DISPLAY_NAME = "Adjust Crop"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "mode": (["white", "transparent", "both"], {"default": "both"}),
            },
        }

    def crop_image(self, image, threshold, padding, mode):
        # Convert to numpy for easier processing
        img_np = image.cpu().numpy()
        batch_size, height, width, channels = img_np.shape
        
        # Create a batch to store the cropped images
        cropped_batch = []
        
        for b in range(batch_size):
            img = img_np[b]
            
            # Create a mask based on the selected mode
            if mode == "white" or mode == "both":
                # Check for white pixels (all channels close to 1.0)
                white_mask = np.all(img > threshold, axis=2)
            
            if mode == "transparent" or mode == "both":
                # Check for transparent pixels (alpha channel close to 0)
                if channels == 4:  # Has alpha channel
                    alpha_mask = img[:, :, 3] < (1.0 - threshold)
                    
                    if mode == "both" and channels == 4:
                        mask = white_mask & ~alpha_mask  # White and not transparent
                    elif mode == "transparent":
                        mask = alpha_mask
                else:
                    # No alpha channel, use only white mask if mode is both
                    mask = white_mask if mode == "both" else np.ones_like(white_mask)
            else:
                # Mode is white only
                mask = white_mask
            
            # Find the bounding box of non-white/non-transparent content
            rows = np.any(~mask, axis=1)
            cols = np.any(~mask, axis=0)
            
            # If the image is completely white/transparent, return the original
            if not np.any(rows) or not np.any(cols):
                cropped_batch.append(img)
                continue
            
            # Find the boundaries
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            rmin = max(0, rmin - padding)
            rmax = min(height - 1, rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(width - 1, cmax + padding)
            
            # Crop the image
            cropped = img[rmin:rmax+1, cmin:cmax+1, :]
            cropped_batch.append(cropped)
        
        # Convert the batch back to a tensor
        # Since the images might have different sizes now, we need to find the max dimensions
        max_h = max(img.shape[0] for img in cropped_batch)
        max_w = max(img.shape[1] for img in cropped_batch)
        
        # Create a new batch with the max dimensions
        result_batch = []
        for cropped in cropped_batch:
            h, w, c = cropped.shape
            # Create a new image with the max dimensions
            new_img = np.zeros((max_h, max_w, channels), dtype=np.float32)
            # Place the cropped image in the top-left corner
            new_img[:h, :w, :] = cropped
            result_batch.append(new_img)
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(np.stack(result_batch)).to(image.device)
        
        return (result_tensor,) 