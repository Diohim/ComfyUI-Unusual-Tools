import torch
import numpy as np

def hex_to_rgb_normalized(hex_color: str) -> tuple[float, float, float]:
    """
    将 #RRGGBB 格式的十六进制颜色字符串转换为归一化 (0.0-1.0) 的 RGB 元组。
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])
    if len(hex_color) != 6:
        print(f"Warning: Invalid hex color '{hex_color}'. Defaulting to black.")
        return (0.0, 0.0, 0.0)
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        print(f"Warning: Could not parse hex color '{hex_color}'. Defaulting to black.")
        return (0.0, 0.0, 0.0)

class FillMaskWithColor:
    """
    一个ComfyUI节点，它使用指定的颜色填充输入图像中被蒙版（白色区域）覆盖的部分。
    蒙版为黑色的区域保持原图不变。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill_color": ("STRING", {
                    "default": "#FFFFFF",
                    "multiline": False,
                }),
            }
        }

    CATEGORY = "mask/compositing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_masked_area"

    def fill_masked_area(self, image: torch.Tensor, mask: torch.Tensor, fill_color: str):
        if mask is None:
            return (image,)
        # 确保输入张量在同一设备和数据类型
        device = image.device
        dtype = image.dtype
        mask = mask.to(device=device, dtype=dtype)
        
        # 1. 将十六进制颜色转换为归一化的RGB张量，作为填充色
        rgb_tuple = hex_to_rgb_normalized(fill_color)
        color_tensor = torch.tensor(rgb_tuple, dtype=dtype, device=device).view(1, 1, 1, 3)

        # 2. 准备填充层
        # 获取图像尺寸 (batch, height, width, channels)
        b, h, w, _ = image.shape
        # 将颜色张量扩展为与输入图像批次和尺寸相匹配的纯色层
        fill_layer = color_tensor.expand(b, h, w, 3)

        # 3. 准备蒙版
        # 蒙版通常是 (B, H, W)，需要扩展为 (B, H, W, 1) 以便与RGB图像进行广播操作
        mask_expanded = mask.unsqueeze(-1)

        # 4. 图像合成 (核心逻辑变更)
        # 公式: result = original_image * (1 - mask) + fill_color * mask
        # 在这里，original_image 是输入图像，fill_color 是填充层
        
        original_image_rgb = image[:, :, :, :3]
        
        # composite_image = (原图 * 反转蒙版) + (填充色 * 蒙版)
        composite_image = original_image_rgb * (1.0 - mask_expanded) + fill_layer * mask_expanded
        
        # 5. 返回结果
        # ComfyUI 的函数需要返回一个元组
        return (composite_image,)

# --- 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "FillMaskWithColor": FillMaskWithColor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillMaskWithColor": "Fill Mask with Color"
}