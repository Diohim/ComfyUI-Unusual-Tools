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
            
            # 添加调试文件，确认节点被加载
            debug_file = os.path.join(self.output_dir, "batch_save_latent_debug.txt")
            with open(debug_file, "w") as f:
                f.write(f"Node initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output directory: {self.output_dir}\n")
                f.write(f"Latents directory: {self.latents_dir}\n")
        except Exception as e:
            print(f"Error creating latents directory: {e}")
            traceback.print_exc()

    # 修改为空返回类型，与官方节点保持一致
    RETURN_TYPES = ()
    FUNCTION = "save_latent_and_image"
    CATEGORY = "latent"
    DISPLAY_NAME = "Batch Save Latent & Image"
    
    # 添加OUTPUT_NODE标记，告诉ComfyUI这是一个输出节点
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "filenames": ("STRING", {"multiline": True, "default": "latent_1\nlatent_2\nlatent_3"}),
                "save_directory": ("STRING", {"default": "latents"}),
            },
            # 添加隐藏参数，与官方节点保持一致
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def save_latent_and_image(self, latent, image, filenames, save_directory, prompt=None, extra_pnginfo=None):
        try:
            # 添加调试文件，确认方法被调用
            debug_file = os.path.join(self.output_dir, "batch_save_latent_execution.txt")
            with open(debug_file, "w") as f:
                f.write(f"Method executed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Latent shape: {latent['samples'].shape}\n")
                f.write(f"Image shape: {image.shape}\n")
                f.write(f"Filenames: {filenames}\n")
                f.write(f"Save directory: {save_directory}\n")
            
            print(f"DEBUG: BatchSaveLatentImage.save_latent_and_image method started!")
            print(f"DEBUG: Filenames: {filenames}")
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

            # 保存的文件路径列表
            saved_files = []
            results = []

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

                    # 准备元数据
                    prompt_info = ""
                    if prompt is not None:
                        prompt_info = json.dumps(prompt)

                    metadata = None
                    try:
                        import args
                        if not args.disable_metadata:
                            metadata = {"prompt": prompt_info}
                            if extra_pnginfo is not None:
                                for x in extra_pnginfo:
                                    metadata[x] = json.dumps(extra_pnginfo[x])
                    except:
                        # 如果args不可用，默认启用元数据
                        metadata = {"prompt": prompt_info}
                        if extra_pnginfo is not None:
                            for x in extra_pnginfo:
                                metadata[x] = json.dumps(extra_pnginfo[x])

                    # 保存 latent
                    latent_filename = f"{filenames[i]}.latent"
                    latent_path = os.path.join(save_dir, latent_filename)
                    print(f"Saving latent to: {latent_path}")
                    
                    latent_saved = False
                    try:
                        # 使用与官方节点相同的格式保存latent
                        output = {}
                        output["latent_tensor"] = latent_sample["samples"]
                        output["latent_format_version_0"] = torch.tensor([])
                        
                        # 方法1: 使用临时文件和torch.save
                        try:
                            temp_path = latent_path + ".tmp"
                            torch.save(output, temp_path)
                            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                if os.path.exists(latent_path):
                                    os.remove(latent_path)
                                os.rename(temp_path, latent_path)
                                print(f"Saved latent using torch.save to: {latent_path}")
                                latent_saved = True
                        except Exception as e1:
                            print(f"Method 1 failed: {e1}")
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                        
                        # 方法2: 如果方法1失败，尝试使用comfy.utils.save_torch_file
                        if not latent_saved:
                            try:
                                import comfy.utils
                                temp_path = latent_path + ".tmp2"
                                comfy.utils.save_torch_file(output, temp_path, metadata=metadata)
                                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                    if os.path.exists(latent_path):
                                        os.remove(latent_path)
                                    os.rename(temp_path, latent_path)
                                    print(f"Saved latent using comfy.utils.save_torch_file to: {latent_path}")
                                    latent_saved = True
                            except Exception as e2:
                                print(f"Method 2 failed: {e2}")
                                if os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                    except:
                                        pass
                        
                        # 方法3: 如果前两种方法都失败，尝试使用原始格式
                        if not latent_saved:
                            try:
                                temp_path = latent_path + ".tmp3"
                                torch.save(latent_sample, temp_path)
                                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                                    if os.path.exists(latent_path):
                                        os.remove(latent_path)
                                    os.rename(temp_path, latent_path)
                                    print(f"Saved latent using original format to: {latent_path}")
                                    latent_saved = True
                            except Exception as e3:
                                print(f"Method 3 failed: {e3}")
                                if os.path.exists(temp_path):
                                    try:
                                        os.remove(temp_path)
                                    except:
                                        pass
                        
                        # 验证文件是否成功保存
                        if latent_saved and os.path.exists(latent_path) and os.path.getsize(latent_path) > 0:
                            print(f"Verified latent file: {latent_path}, size: {os.path.getsize(latent_path)} bytes")
                            saved_files.append(latent_path)
                            
                            # 添加到结果列表，用于UI显示
                            results.append({
                                "filename": latent_filename,
                                "subfolder": os.path.relpath(save_dir, self.output_dir) if not os.path.isabs(save_dir) else save_dir,
                                "type": "output"
                            })
                        else:
                            raise Exception("Failed to save latent file after all attempts")
                            
                    except Exception as e:
                        print(f"Error saving latent to {latent_path}: {e}")
                        traceback.print_exc()

                    # 获取图像 (处理批量大小差异)
                    image_idx = min(i, image_batch_size - 1)
                    img = image[image_idx]

                    # 保存图像
                    img_filename = f"{filenames[i]}.png"
                    img_path = os.path.join(save_dir, img_filename)
                    print(f"Saving image to: {img_path}") # 增加保存前的打印
                    try:
                        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                        Image.fromarray(img_np).save(img_path)
                        print(f"Saved image to: {img_path}")
                        saved_files.append(img_path)
                        
                        # 添加到结果列表，用于UI显示
                        results.append({
                            "filename": img_filename,
                            "subfolder": os.path.relpath(save_dir, self.output_dir) if not os.path.isabs(save_dir) else save_dir,
                            "type": "output"
                        })
                    except Exception as e:
                        print(f"Error saving image to {img_path}: {e}")
                        traceback.print_exc()

                    # 保存元数据以链接 latent 和图像
                    metadata_json = {
                        "latent_file": f"{filenames[i]}.latent",
                        "image_file": f"{filenames[i]}.png",
                    }
                    metadata_filename = f"{filenames[i]}.json"
                    metadata_path = os.path.join(save_dir, metadata_filename)
                    print(f"Saving metadata to: {metadata_path}") # 增加保存前的打印
                    try:
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata_json, f)
                        print(f"Saved metadata to: {metadata_path}")
                        saved_files.append(metadata_path)
                    except Exception as e:
                        print(f"Error saving metadata to {metadata_path}: {e}")
                        traceback.print_exc()
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    traceback.print_exc()

            print(f"Saved {max(latent_batch_size, image_batch_size)} latent(s) and image(s) to {save_dir}")
            print(f"DEBUG: BatchSaveLatentImage.save_latent_and_image method completed!")
            
            # 更新调试文件，确认方法执行完成
            with open(debug_file, "a") as f:
                f.write(f"Method completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Saved {max(latent_batch_size, image_batch_size)} latent(s) and image(s) to {save_dir}\n")
                f.write(f"Saved files: {saved_files}\n")

            # 返回UI显示信息，与官方节点保持一致
            return {"ui": {"latents": results}}

        except Exception as e:
            print(f"Error in save_latent_and_image: {e}")
            traceback.print_exc()
            
            # 记录错误到调试文件
            debug_file = os.path.join(self.output_dir, "batch_save_latent_error.txt")
            with open(debug_file, "w") as f:
                f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")

            # 返回空UI信息
            return {"ui": {"latents": []}}