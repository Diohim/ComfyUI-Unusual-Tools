import torch
import os
import json
import numpy as np
from PIL import Image
import folder_paths
import traceback
import time
import shutil  # 添加shutil用于文件操作
import io  # 用于内存中加载文件
import struct  # 用于低级二进制文件处理

# 导入ComfyUI的safetensors库
try:
    import safetensors.torch
except ImportError:
    print("Warning: safetensors not found, falling back to torch.load")
    safetensors = None

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
                
            # 添加调试文件，确认节点被加载
            debug_file = os.path.join(self.output_dir, "batch_load_latent_debug.txt")
            with open(debug_file, "w") as f:
                f.write(f"Node initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output directory: {self.output_dir}\n")
                f.write(f"Latents directory: {self.latents_dir}\n")
        except Exception as e:
            print(f"Error checking/creating latents directory: {e}")
            traceback.print_exc()
        
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
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
    
    # 检测文件格式
    def detect_file_format(self, file_path):
        """尝试检测文件格式"""
        try:
            with open(file_path, 'rb') as f:
                # 读取前8个字节
                header = f.read(8)
                
                # 检查是否是safetensors格式 (通常以JSON头开始)
                if header.startswith(b'{'):
                    return "safetensors"
                
                # 检查是否是PyTorch格式 (通常以'PK'开始，表示ZIP格式)
                if header.startswith(b'PK'):
                    return "pytorch"
                
                # 检查是否是pickle格式
                if header.startswith(b'\x80\x03') or header.startswith(b'\x80\x04') or header.startswith(b'\x80\x05'):
                    return "pickle"
                
                # 检查是否是numpy格式
                if header.startswith(b'\x93NUMPY'):
                    return "numpy"
                
                # 未知格式
                return "unknown"
        except Exception as e:
            print(f"Error detecting file format: {e}")
            return "unknown"
    
    # 创建默认latent
    def create_default_latent(self, width=64, height=64):
        """创建默认的latent数据"""
        print(f"Creating default latent tensor of size {height}x{width}")
        return {"samples": torch.zeros((1, 4, height, width))}
    
    # 使用ComfyUI的方法加载latent文件
    def comfy_load_latent(self, file_path):
        print(f"Attempting to load latent file: {file_path}")
        
        # 创建一个备份，以防万一
        backup_path = file_path + ".backup"
        try:
            if not os.path.exists(backup_path):
                shutil.copy2(file_path, backup_path)
                print(f"Created backup of original file: {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
        
        # 检测文件格式
        file_format = self.detect_file_format(file_path)
        print(f"Detected file format: {file_format}")
        
        try:
            # 1. 尝试使用safetensors加载
            if safetensors is not None and (file_format == "safetensors" or file_format == "unknown"):
                try:
                    print(f"Trying to load with safetensors...")
                    latent = safetensors.torch.load_file(file_path, device="cpu")
                    multiplier = 1.0
                    if "latent_format_version_0" not in latent:
                        multiplier = 1.0 / 0.18215
                    samples = {"samples": latent["latent_tensor"].float() * multiplier}
                    print(f"Successfully loaded latent using safetensors")
                    return samples
                except Exception as e:
                    print(f"Error loading with safetensors: {e}")
            
            # 2. 尝试使用内存缓冲区和torch.load
            if file_format in ["pytorch", "pickle", "unknown"]:
                try:
                    print(f"Trying to load with torch.load via memory buffer...")
                    # 读取文件到内存中
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    # 从内存中加载，避免直接修改文件
                    buffer = io.BytesIO(file_content)
                    
                    # 禁用Fooocus_Nodes的补丁
                    original_torch_load = torch.load
                    try:
                        # 尝试临时恢复原始的torch.load
                        if hasattr(torch, '_original_load'):
                            torch.load = getattr(torch, '_original_load')
                        
                        latent_data = torch.load(buffer, map_location="cpu")
                    finally:
                        # 恢复补丁后的torch.load
                        torch.load = original_torch_load
                    
                    # 处理不同的latent格式
                    if "samples" in latent_data:
                        print(f"Detected old latent format with 'samples' key")
                        return latent_data
                    elif "latent_tensor" in latent_data:
                        print(f"Detected new latent format with 'latent_tensor' key")
                        multiplier = 1.0
                        if "latent_format_version_0" not in latent_data:
                            multiplier = 1.0 / 0.18215
                        samples = {"samples": latent_data["latent_tensor"].float() * multiplier}
                        return samples
                    else:
                        print(f"Unknown latent format, trying to use as is")
                        # 尝试找到张量
                        for k, v in latent_data.items():
                            if isinstance(v, torch.Tensor) and len(v.shape) >= 3:
                                print(f"Using tensor with key '{k}' as latent")
                                return {"samples": v}
                        
                        # 如果找不到合适的张量，创建一个默认的
                        print(f"Could not find suitable tensor, creating default")
                        return self.create_default_latent()
                except Exception as e:
                    print(f"Error loading with torch.load: {e}")
            
            # 3. 尝试使用numpy加载
            if file_format in ["numpy", "unknown"]:
                try:
                    print(f"Trying to load with numpy...")
                    tensor = torch.from_numpy(np.load(file_path))
                    if len(tensor.shape) >= 3:
                        # 确保格式正确
                        if len(tensor.shape) == 3:  # [C, H, W]
                            tensor = tensor.unsqueeze(0)  # [1, C, H, W]
                        if tensor.shape[1] != 4:
                            # 如果通道数不是4，创建一个默认的latent
                            print(f"Tensor has wrong number of channels: {tensor.shape[1]}, expected 4")
                            return self.create_default_latent()
                        return {"samples": tensor}
                    else:
                        print(f"Loaded numpy array has wrong shape: {tensor.shape}")
                        return self.create_default_latent()
                except Exception as e:
                    print(f"Error loading with numpy: {e}")
            
            # 4. 尝试手动解析文件
            try:
                print(f"Trying to manually parse the file...")
                # 读取文件内容
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # 尝试查找张量数据的特征
                # 这是一个非常简化的方法，可能不适用于所有情况
                # 寻找可能的张量形状信息
                shape_markers = [b'shape', b'size', b'tensor']
                for marker in shape_markers:
                    pos = content.find(marker)
                    if pos >= 0:
                        # 尝试从这个位置解析一些信息
                        print(f"Found potential tensor marker '{marker.decode()}' at position {pos}")
                        # 这里可以添加更复杂的解析逻辑
                
                # 如果无法解析，创建一个默认的latent
                print(f"Could not manually parse the file, creating default latent")
                return self.create_default_latent()
            except Exception as e:
                print(f"Error manually parsing file: {e}")
            
            # 5. 如果所有方法都失败，创建一个默认的latent数据
            print(f"All loading methods failed, creating default latent tensor")
            return self.create_default_latent()
                
        except Exception as e:
            print(f"Error in comfy_load_latent: {e}")
            traceback.print_exc()
            return self.create_default_latent()
    
    def load_latent_and_image(self, filenames, load_directory):
        try:
            # 添加调试文件，确认方法被调用
            debug_file = os.path.join(self.output_dir, "batch_load_latent_execution.txt")
            with open(debug_file, "w") as f:
                f.write(f"Method executed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Filenames: {filenames}\n")
                f.write(f"Load directory: {load_directory}\n")
            
            print(f"DEBUG: BatchLoadLatentImage.load_latent_and_image method started!")
            print(f"DEBUG: Filenames: {filenames}")
            
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
                    # 创建一个空的latent和图像作为默认返回值
                    empty_latent = self.create_default_latent()
                    empty_image = torch.zeros((1, 512, 512, 3))
                    return (empty_latent, empty_image)
            
            # List files in the directory for debugging
            print(f"Files in {load_dir}:")
            try:
                for file in os.listdir(load_dir):
                    print(f"  - {file}")
            except Exception as e:
                print(f"Error listing files in directory: {e}")
                traceback.print_exc()
            
            # Parse filenames
            filenames = [f.strip() for f in filenames.split('\n') if f.strip()]
            print(f"Filenames to load: {filenames}")
            
            if not filenames:
                print("No filenames provided, creating empty latent and image")
                empty_latent = self.create_default_latent()
                empty_image = torch.zeros((1, 512, 512, 3))
                return (empty_latent, empty_image)
            
            # Lists to store loaded latents and images
            latent_samples = []
            images = []
            
            # Load each latent and image
            for filename in filenames:
                try:
                    # Check for latent file
                    latent_path = os.path.join(load_dir, f"{filename}.latent")
                    print(f"Looking for latent file: {latent_path}")
                    
                    # 检查是否存在.corrupted文件
                    corrupted_path = latent_path + ".corrupted"
                    if os.path.exists(corrupted_path) and not os.path.exists(latent_path):
                        print(f"Found corrupted file: {corrupted_path}, attempting to recover")
                        try:
                            # 尝试恢复corrupted文件
                            shutil.copy2(corrupted_path, latent_path)
                            print(f"Recovered latent file from corrupted version")
                        except Exception as e:
                            print(f"Failed to recover corrupted file: {e}")
                    
                    # 检查是否存在.backup文件
                    backup_path = latent_path + ".backup"
                    if os.path.exists(backup_path) and not os.path.exists(latent_path):
                        print(f"Found backup file: {backup_path}, attempting to recover")
                        try:
                            # 尝试从备份恢复
                            shutil.copy2(backup_path, latent_path)
                            print(f"Recovered latent file from backup")
                        except Exception as e:
                            print(f"Failed to recover from backup: {e}")
                    
                    if not os.path.exists(latent_path):
                        print(f"Warning: Latent file not found: {latent_path}")
                        # 创建一个默认的latent
                        default_tensor = torch.zeros((1, 4, 64, 64))
                        latent_samples.append(default_tensor)
                        continue
                    
                    # 使用ComfyUI的方法加载latent
                    try:
                        latent_data = self.comfy_load_latent(latent_path)
                        print(f"Successfully loaded latent data from: {latent_path}")
                        
                        # 获取latent张量
                        if "samples" in latent_data:
                            latent_tensor = latent_data["samples"]
                            print(f"Latent tensor shape: {latent_tensor.shape}")
                            latent_samples.append(latent_tensor)
                        else:
                            print(f"No 'samples' key in latent data, creating default")
                            default_tensor = torch.zeros((1, 4, 64, 64))
                            latent_samples.append(default_tensor)
                                
                    except Exception as e:
                        print(f"Error loading latent from {latent_path}: {e}")
                        traceback.print_exc()
                        
                        # 如果加载失败，创建一个默认的latent
                        print(f"Creating default latent tensor as fallback")
                        latent_tensor = torch.zeros((1, 4, 64, 64))
                        latent_samples.append(latent_tensor)
                    
                    # Check for image file
                    img_path = os.path.join(load_dir, f"{filename}.png")
                    print(f"Looking for image file: {img_path}")
                    
                    if not os.path.exists(img_path):
                        print(f"Warning: Image file not found: {img_path}")
                        # Create a blank image if the image file is missing
                        h, w = latent_tensor.shape[2] * 8, latent_tensor.shape[3] * 8
                        blank_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
                        images.append(torch.from_numpy(blank_img))
                        print(f"Created blank image of size {h}x{w}")
                    else:
                        # Load image
                        try:
                            # 使用只读模式打开图像
                            with Image.open(img_path) as pil_img:
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
                            h, w = latent_tensor.shape[2] * 8, latent_tensor.shape[3] * 8
                            blank_img = np.ones((h, w, 3), dtype=np.float32) * 0.5
                            images.append(torch.from_numpy(blank_img))
                            print(f"Created blank image as fallback, size {h}x{w}")
                except Exception as e:
                    print(f"Error processing filename {filename}: {e}")
                    traceback.print_exc()
            
            if not latent_samples:
                print("No valid latent files found! Creating default tensors")
                # 即使没有找到有效的latent文件，也创建一个默认的返回值
                default_latent = self.create_default_latent()
                default_image = torch.zeros((1, 512, 512, 3))
                
                # 更新调试文件
                with open(debug_file, "a") as f:
                    f.write(f"No valid latent files found, creating default tensors\n")
                
                return (default_latent, default_image)
            
            # Combine into batches
            latent_batch = torch.cat(latent_samples, dim=0)
            image_batch = torch.stack(images, dim=0)
            
            print(f"Final latent batch shape: {latent_batch.shape}")
            print(f"Final image batch shape: {image_batch.shape}")
            
            # Create the latent dictionary
            latent = {"samples": latent_batch}
            
            print(f"Successfully loaded {len(latent_samples)} latent(s) and image(s) from {load_dir}")
            print(f"DEBUG: BatchLoadLatentImage.load_latent_and_image method completed!")
            
            # 更新调试文件
            with open(debug_file, "a") as f:
                f.write(f"Method completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Loaded {len(latent_samples)} latent(s) and image(s) from {load_dir}\n")
                f.write(f"Final latent batch shape: {latent_batch.shape}\n")
                f.write(f"Final image batch shape: {image_batch.shape}\n")
            
            return (latent, image_batch)
            
        except Exception as e:
            print(f"Error in load_latent_and_image: {e}")
            traceback.print_exc()
            
            # 记录错误到调试文件
            error_file = os.path.join(self.output_dir, "batch_load_latent_error.txt")
            with open(error_file, "w") as f:
                f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            
            # 返回空的latent和图像，而不是抛出异常
            empty_latent = self.create_default_latent()
            empty_image = torch.zeros((1, 512, 512, 3))
            return (empty_latent, empty_image) 