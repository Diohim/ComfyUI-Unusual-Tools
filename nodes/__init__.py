from .auto_image_resize import AutoImageResize
from .adjust_crop import AdjustCrop
from .batch_save_latent_image import BatchSaveLatentImage
from .batch_load_latent_image import BatchLoadLatentImage

NODE_CLASS_MAPPINGS = {
    "AutoImageResize": AutoImageResize,
    "AdjustCrop": AdjustCrop,
    "BatchSaveLatentImage": BatchSaveLatentImage,
    "BatchLoadLatentImage": BatchLoadLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoImageResize": "Auto Image Resize",
    "AdjustCrop": "Adjust Crop",
    "BatchSaveLatentImage": "Batch Save Latent & Image",
    "BatchLoadLatentImage": "Batch Load Latent & Image",
} 