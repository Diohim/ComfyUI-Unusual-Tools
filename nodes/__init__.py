from .auto_image_resize import AutoImageResize
from .adjust_crop import AdjustCrop
from .batch_save_latent_image import BatchSaveLatentImage
from .batch_load_latent_image import BatchLoadLatentImage
from .fill_mask_with_color import FillMaskWithColor

NODE_CLASS_MAPPINGS = {
    "AutoImageResize": AutoImageResize,
    "AdjustCrop": AdjustCrop,
    "BatchSaveLatentImage": BatchSaveLatentImage,
    "BatchLoadLatentImage": BatchLoadLatentImage,
    "FillMaskWithColor": FillMaskWithColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoImageResize": "Auto Image Resize",
    "AdjustCrop": "Adjust Crop",
    "BatchSaveLatentImage": "Batch Save Latent & Image",
    "BatchLoadLatentImage": "Batch Load Latent & Image",
    "FillMaskWithColor": "Fill Mask With Color",
} 