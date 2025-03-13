import torch

class AutoImageResize:
    def __init__(self):
        pass

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "resize_and_expand_image"
    CATEGORY = "image/transform"
    DISPLAY_NAME = "Auto Image Resize"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 8192, "step": 1}),
            },
        }

    def resize_and_expand_image(self, image, target_height, target_width, feathering):
        resized, top, bottom, left, right = self.resize(image, target_height, target_width)
        return self.expand_image(resized, left, top, right, bottom, feathering)

    def resize(self, image, target_height, target_width):
        height, width = image.shape[1:3]
        
        if width/height > target_width/target_height:
            # Scale width to target, calculate scaled height
            scaled_height = int(height/width*target_width)
            top = (target_height-scaled_height)//2
            bottom = target_height-scaled_height-top
            resized = torch.nn.functional.interpolate(
                image.movedim(-1, 1),  
                size=(scaled_height, target_width), 
                mode="bicubic",
                antialias=True
            ).movedim(1, -1).clamp(0.0, 1.0)
            
            return resized, top, bottom, 0, 0
        else:
            # Scale height to target, calculate scaled width
            scaled_width = int(width/height*target_height)
            left = (target_width-scaled_width)//2
            right = target_width-scaled_width-left
            resized = torch.nn.functional.interpolate(
                image.movedim(-1, 1),  
                size=(target_height, scaled_width), 
                mode="bicubic",
                antialias=True
            ).movedim(1, -1).clamp(0.0, 1.0)
            
            return resized, 0, 0, left, right

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.size()

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:
            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering
                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask) 