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
            # 确定保存目录
            if save_directory.strip() == "" or save_directory.strip() == "latents":
                save_dir = self.latents_dir
                print(f"Using default latents directory: {save_dir}")
            else:
                # 检查是否为绝对路径
                if os.path.isabs(save_directory):
                    save_dir = save_directory
                else:
                    # 如果是相对路径，则相对于输出目录
                    save_dir = os.path.join(self.output_dir, save_directory)

                try:
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Custom directory created/verified: {save_dir}")
                except Exception as e:
                    print(f"Error creating custom directory: {e}")
                    traceback.print_exc()
                    # 回退到默认目录
                    save_dir = self.latents_dir
                    print(f"Falling back to default directory: {save_dir}")

            # 解析文件名，去除空行和首尾空格
            filenames = [f.strip() for f in filenames.split('\n') if f.strip()]

            # 获取批量大小
            latent_batch_size = latent["samples"].shape[0]
            image_batch_size = image.shape[0]

            print(f"Latent batch size: {latent_batch_size}, Image batch size: {image_batch_size}")

            # 确定是否为批量模式 (只要latent或image有一个是批量，就认为是批量)
            is_batch = latent_batch_size > 1 or image_batch_size > 1

            # 如果没有提供文件名，则生成默认文件名
            if not filenames:
                timestamp = int(time.time())
                filenames = [f"latent_{timestamp}_{i}" for i in range(max(latent_batch_size, image_batch_size))]
                print(f"Generated default filenames: {filenames}")
            elif len(filenames) < max(latent_batch_size, image_batch_size): # 确保文件名数量足够，如果不足则复制最后一个文件名补齐
                last_filename = filenames[-1] if filenames else "latent_default" # 避免filenames为空的情况
                filenames.extend([f"{last_filename}_{i+len(filenames)}" for i in range(max(latent_batch_size, image_batch_size) - len(filenames))])
                print(f"Filenames 부족, added filenames: {filenames}")
            elif len(filenames) > max(latent_batch_size, image_batch_size): # 如果文件名数量过多，则截断
                filenames = filenames[:max(latent_batch_size, image_batch_size)]
                print(f"Filenames 과다, truncated filenames: {filenames}")


            # 保存每个latent和图像
            for i in range(max(latent_batch_size, image_batch_size)):
                try:
                    # 获取 latent sample (处理批量大小差异)
                    latent_idx = min(i, latent_batch_size - 1)
                    latent_sample = {
                        "samples": latent["samples"][latent_idx:latent_idx+1], # 确保切片是 [idx:idx+1] 保持维度
                    }
                    if "noise_mask" in latent:
                        if latent["noise_mask"] is not None:
                            # 修复 noise_mask 的索引问题，确保正确处理批量和非批量 noise_mask
                            if latent["noise_mask"].shape[0] > 1: # 如果 noise_mask 也是批量的
                                latent_sample["noise_mask"] = latent["noise_mask"][latent_idx:latent_idx+1]
                            else: # 如果 noise_mask 不是批量的，则直接使用
                                latent_sample["noise_mask"] = latent["noise_mask"]

                    # 保存 latent
                    latent_path = os.path.join(save_dir, f"{filenames[i]}.latent")
                    print(f"Saving latent to: {latent_path}") # 增加保存前的打印
                    try:
                        torch.save(latent_sample, latent_path)
                        print(f"Saved latent to: {latent_path}")
                    except Exception as e:
                        print(f"Error saving latent to {latent_path}: {e}")
                        traceback.print_exc()

                    # 获取图像 (处理批量大小差异)
                    image_idx = min(i, image_batch_size - 1)
                    img = image[image_idx]

                    # 保存图像
                    img_path = os.path.join(save_dir, f"{filenames[i]}.png")
                    print(f"Saving image to: {img_path}") # 增加保存前的打印
                    try:
                        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(img_path)
                        print(f"Saved image to: {img_path}")
                    except Exception as e:
                        print(f"Error saving image to {img_path}: {e}")
                        traceback.print_exc()

                    # 保存元数据以链接 latent 和图像
                    metadata = {
                        "latent_file": f"{filenames[i]}.latent",
                        "image_file": f"{filenames[i]}.png",
                    }
                    metadata_path = os.path.join(save_dir, f"{filenames[i]}.json")
                    print(f"Saving metadata to: {metadata_path}") # 增加保存前的打印
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

        # 没有返回值
        return ()