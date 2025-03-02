import os
import sys
import requests  # Add this
import torch
import time
import numpy as np
import cv2
from PIL import Image
import folder_paths
import comfy.model_management
from typing import List, Optional, Dict, Any, Union, Tuple
from tqdm import tqdm  # Add this
from transformers import AutoModelForCausalLM
from janus.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)

# Get device
device = comfy.model_management.get_torch_device()

class ModelDownloader:
    """Handles model downloading and verification"""
    
    MODEL_FILES = {
        'config.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/config.json',
        'preprocessor_config.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/preprocessor_config.json',
        'processor_config.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/processor_config.json',
        'pytorch_model.bin.index.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/pytorch_model.bin.index.json',
        'special_tokens_map.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/special_tokens_map.json',
        'tokenizer.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/tokenizer.json',
        'tokenizer_config.json': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/raw/main/tokenizer_config.json',
        'pytorch_model-00001-of-00002.bin': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/resolve/main/pytorch_model-00001-of-00002.bin',
        'pytorch_model-00002-of-00002.bin': 'https://huggingface.co/deepseek-ai/Janus-Pro-7B/resolve/main/pytorch_model-00002-of-00002.bin'
    }

    @staticmethod
    def download_file(url: str, filepath: str, desc: str = None) -> None:
        """Download a file with progress bar"""
        print(f"Downloading from {url} to {filepath}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(filepath, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                pbar.update(size)

    @staticmethod
    def ensure_model_downloaded(model_path: str) -> None:
        """Ensure all model files are present, downloading if needed"""
        print(f"Creating model directory if it doesn't exist: {model_path}")
        os.makedirs(model_path, exist_ok=True)
        
        missing_files = []
        for filename in ModelDownloader.MODEL_FILES.keys():
            filepath = os.path.join(model_path, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            print(f"Found {len(missing_files)} missing files. Starting download...")
            for filename in missing_files:
                url = ModelDownloader.MODEL_FILES[filename]
                filepath = os.path.join(model_path, filename)
                print(f"\nDownloading {filename} from {url}")
                try:
                    ModelDownloader.download_file(url, filepath, desc=f"Downloading {filename}")
                    print(f"Successfully downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {str(e)}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise Exception(f"Failed to download {filename}: {str(e)}")
        else:
            print("All model files are present.")

    @staticmethod
    def download_all_files(model_path: str) -> None:
        """Download all model files (force download)"""
        print(f"Starting download of all model files to {model_path}")
        os.makedirs(model_path, exist_ok=True)
        
        for filename, url in ModelDownloader.MODEL_FILES.items():
            filepath = os.path.join(model_path, filename)
            print(f"Downloading {filename}...")
            try:
                ModelDownloader.download_file(url, filepath)
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise

    @staticmethod
    def verify_model_files(model_path: str) -> bool:
        """Verify all required model files exist"""
        print(f"Verifying model files in {model_path}")
        missing = []
        for filename in ModelDownloader.MODEL_FILES.keys():
            filepath = os.path.join(model_path, filename)
            if not os.path.exists(filepath):
                missing.append(filename)
        
        if missing:
            print(f"Missing files: {', '.join(missing)}")
            return False
        print("All model files verified successfully")
        return True

    @staticmethod
    def ensure_model_downloaded(model_path: str) -> None:
        """Ensure all model files are present, downloading if needed"""
        os.makedirs(model_path, exist_ok=True)
        
        print(f"Checking Janus-Pro model files in {model_path}")
        for filename, url in ModelDownloader.MODEL_FILES.items():
            filepath = os.path.join(model_path, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                try:
                    ModelDownloader.download_file(url, filepath, desc=f"Downloading {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {str(e)}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise e

    @staticmethod
    def verify_model_files(model_path: str) -> bool:
        """Verify all required model files exist"""
        return all(
            os.path.exists(os.path.join(model_path, filename))
            for filename in ModelDownloader.MODEL_FILES.keys()
        )

# Import necessary Janus components
from janus.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)

# Get device
device = comfy.model_management.get_torch_device()

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def resize_maintain_aspect(img: np.ndarray, target_size: int, target_dim: str = 'width') -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = img.shape[:2]
        if target_dim == 'width':
            aspect = h / w
            new_w = target_size
            new_h = int(aspect * new_w)
        else:
            aspect = w / h
            new_h = target_size
            new_w = int(aspect * new_h)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def get_aspect_ratio(width: int, height: int) -> float:
        """Calculate aspect ratio"""
        return width / height

    @staticmethod
    def add_frame(image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Add frame around image"""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w-1, h-1), color, thickness)
        return image

    @staticmethod
    def process_images(first_image: torch.Tensor, second_image: Optional[torch.Tensor] = None, 
                      target_size: int = 1024, frame_thickness: int = 2) -> Tuple[List[Image.Image], str]:
        """Process and combine images if needed"""
        # Handle single image
        if second_image is None:
            first_image = (first_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
            first_image = ImageProcessor.add_frame(first_image.copy(), thickness=frame_thickness)
            h, w = first_image.shape[:2]
            ratio = w / h
            
            if ratio > 1:
                new_width = target_size
                new_height = int(target_size / ratio)
            else:
                new_height = target_size
                new_width = int(target_size * ratio)
                
            first_image = cv2.resize(first_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        #   first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
            return [Image.fromarray(first_image)], "single"

        # Handle two images
        first_image = (first_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        second_image = (second_image[0].detach().cpu().numpy() * 255).astype(np.uint8)
        
        # Determine layout
        h1, w1 = first_image.shape[:2]
        h2, w2 = second_image.shape[:2]
        horiz_ratio = ImageProcessor.get_aspect_ratio(w1 + w2, max(h1, h2))
        vert_ratio = ImageProcessor.get_aspect_ratio(max(w1, w2), h1 + h2)
        use_horizontal = abs(horiz_ratio - 1.33) < abs(vert_ratio - 1.33)
        
        if use_horizontal:
            target_height = min(h1, h2)
            first_image = ImageProcessor.resize_maintain_aspect(first_image, target_height, 'height')
            second_image = ImageProcessor.resize_maintain_aspect(second_image, target_height, 'height')
            h1, w1 = first_image.shape[:2]
            h2, w2 = second_image.shape[:2]
            
            first_image = ImageProcessor.add_frame(first_image.copy(), thickness=frame_thickness)
            second_image = ImageProcessor.add_frame(second_image.copy(), thickness=frame_thickness)
            
            combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            combined[:h1, :w1] = first_image
            combined[:h2, w1:w1+w2] = second_image
            layout = "horizontal"
        else:
            target_width = min(w1, w2)
            first_image = ImageProcessor.resize_maintain_aspect(first_image, target_width, 'width')
            second_image = ImageProcessor.resize_maintain_aspect(second_image, target_width, 'width')
            h1, w1 = first_image.shape[:2]
            h2, w2 = second_image.shape[:2]
            
            first_image = ImageProcessor.add_frame(first_image.copy(), thickness=frame_thickness)
            second_image = ImageProcessor.add_frame(second_image.copy(), thickness=frame_thickness)
            
            combined = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
            combined[:h1, :w1] = first_image
            combined[h1:h1+h2, :w2] = second_image
            layout = "vertical"

        # Final resize
        final_ratio = ImageProcessor.get_aspect_ratio(combined.shape[1], combined.shape[0])
        if final_ratio > 1:
            new_width = target_size
            new_height = int(target_size / final_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * final_ratio)
            
        combined = cv2.resize(combined, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        return [Image.fromarray(combined)], layout

class ModelManager:
    """Utility class for model operations"""
    
    @staticmethod
    def get_generation_config(processor: Any, inputs_embeds: torch.Tensor, 
                            attention_mask: torch.Tensor, max_new_tokens: int,
                            temperature: float, top_p: float) -> Dict[str, Any]:
        """Create generation configuration"""
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "bos_token_id": processor.tokenizer.bos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "use_cache": True,
        }

    @staticmethod
    def is_valid_model(model_dir: str) -> bool:
        """Validate model directory"""
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        
        if not all(os.path.exists(os.path.join(model_dir, file)) for file in required_files):
            return False
            
        if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            return True
            
        # Check for sharded model files
        bin_files = [f for f in os.listdir(model_dir) 
                    if f.startswith("pytorch_model-") and f.endswith(".bin")]
        return len(bin_files) > 0

class ChatManager:
    """Utility class for chat history management"""
    
    @staticmethod
    def format_chat_history(history: List[Dict[str, Any]]) -> str:
        """Format chat history for display"""
        try:
            formatted = []
            for entry in history:
                timestamp = entry.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                role = entry.get("role", "").replace("<|", "").replace("|>", "")
                content = entry.get("content", "")
                
                # Clean up content - remove image placeholder if present
                content = content.replace("<image_placeholder>\n", "")
                
                entry_text = f"[{timestamp}] {role}:\n{content}\n"
                if "images" in entry:
                    entry_text += "[Image Included]\n"
                entry_text += "-" * 50 + "\n"
                
                formatted.append(entry_text)
            
            return "\n".join(formatted)
        except Exception as e:
            print(f"Error formatting history: {str(e)}")
            return str(history)

    @staticmethod
    def build_prompt(prompt: str, layout: str = "single") -> str:
        """Build conversation prompt"""
        if layout == "single":
            return prompt
        
        first_pos = "left" if layout == "horizontal" else "top"
        second_pos = "right" if layout == "horizontal" else "bottom"
        return f"""The first image is on the {first_pos}, the second image is on the {second_pos}.

{prompt}"""