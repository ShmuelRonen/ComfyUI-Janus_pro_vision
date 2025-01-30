from typing import List, Optional, Dict, Any, Union, Tuple
import os
import torch
import time
import folder_paths
import comfy.model_management
from transformers import AutoModelForCausalLM
from .janus.models import VLChatProcessor
from .utils import ImageProcessor, ModelManager, ChatManager, ModelDownloader  # Add ModelDownloader here

# Get device
device = comfy.model_management.get_torch_device()

def load_model(self, model_path="base"):
    """Load the model"""
    try:
        if self.loaded_path != model_path or self.model is None or self.processor is None:
            model_folder = janus_model_path
            print(f"Checking model folder: {model_folder}")
            
            # Always try to download if any file is missing
            try:
                print("Checking for missing files and downloading if needed...")
                ModelDownloader.ensure_model_downloaded(model_folder)
            except Exception as e:
                print(f"Download error: {str(e)}")
                return ({"error": f"Model download failed: {str(e)}", "model": None, "processor": None},)

            # Verify after download attempt
            if not ModelDownloader.verify_model_files(model_folder):
                error_msg = "Model files are still incomplete after download attempt"
                print(error_msg)
                return ({"error": error_msg, "model": None, "processor": None},)

            print(f"Loading model from: {model_folder}...")
            
            try:
                self.processor = VLChatProcessor.from_pretrained(model_folder)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_folder,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    use_safetensors=False,
                    low_cpu_mem_usage=True,
                )
                self.model = self.model.to(device).eval()
            except Exception as e:
                error_msg = f"Error loading model: {str(e)}"
                print(error_msg)
                return ({"error": error_msg, "model": None, "processor": None},)
            
            self.loaded_path = model_path
            print("Model loaded successfully")
            
        return ({"model": self.model, "processor": self.processor},)
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        print(error_msg)
        return ({"error": error_msg, "model": None, "processor": None},)

# Get device
device = comfy.model_management.get_torch_device()

# Set up model paths
models_path = folder_paths.models_dir
janus_model_path = os.path.join(models_path, "Janus-Pro")
os.makedirs(janus_model_path, exist_ok=True)
folder_paths.add_model_folder_path("janus_model", janus_model_path)

class VisionModelLoader:
    """Universal Janus model loader"""
    
    RETURN_TYPES = ("JANUS_MODEL",)
    RETURN_NAMES = ("janus_model",)
    FUNCTION = "load_model"
    CATEGORY = "JanusVision"

    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded_path = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (["base"],),
            }
        }

    def load_model(self, model_path="base"):
        """Load the model"""
        try:
            model_folder = janus_model_path
            
            # Check if any required file is missing
            missing_files = False
            for filename in ModelDownloader.MODEL_FILES.keys():
                if not os.path.exists(os.path.join(model_folder, filename)):
                    missing_files = True
                    break
            
            # If any file is missing, download all files
            if missing_files:
                print("Model files missing. Starting download...")
                try:
                    ModelDownloader.download_all_files(model_folder)
                except Exception as e:
                    return ({"error": f"Download failed: {str(e)}", "model": None, "processor": None},)
            
            print(f"Loading model from: {model_folder}")
            
            # Load the model
            try:
                self.processor = VLChatProcessor.from_pretrained(model_folder)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_folder,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device,
                    use_safetensors=False,
                    low_cpu_mem_usage=True,
                )
                self.model = self.model.to(device).eval()
                self.loaded_path = model_path
                print("Model loaded successfully")
                
            except Exception as e:
                return ({"error": f"Model loading failed: {str(e)}", "model": None, "processor": None},)
            
            return ({"model": self.model, "processor": self.processor},)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return ({"error": error_msg, "model": None, "processor": None},)

class UnifiedVisionAnalyzer:
    """Unified image analysis with optional chat functionality"""
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("response", "chat_history")
    FUNCTION = "analyze"
    CATEGORY = "JanusVision"

    def __init__(self):
        self.chat_history = []

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "janus_model": ("JANUS_MODEL",),
                "image_a": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Please describe this image.",
                    "placeholder": "Enter your prompt"
                }),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "image_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "frame_size": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "reset_chat": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_b": ("IMAGE",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def analyze(self, janus_model, image_a, prompt, chat_mode=False,
                seed=42, temperature=0.1, top_p=0.95, max_tokens=512,
                image_size=1024, frame_size=2, reset_chat=False, image_b=None):
        try:
            # Handle chat reset
            if reset_chat:
                self.chat_history = []
                return ("Chat history cleared.", "")

            # Validate model
            if isinstance(janus_model, dict) and "error" in janus_model:
                return (f"Error: {janus_model['error']}", "")
            
            model = janus_model.get("model") if isinstance(janus_model, dict) else janus_model
            processor = janus_model.get("processor") if isinstance(janus_model, dict) else getattr(janus_model, "processor", None)
            
            if model is None or processor is None:
                return ("Error: Model or processor not properly loaded", "")

            # Set random seed
            if seed != -1:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            
            # Process images and build prompt
            images, layout = ImageProcessor.process_images(image_a, image_b, image_size, frame_size)
            prompt_text = ChatManager.build_prompt(prompt, layout)

            # Build current conversation
            current_turn = {
                "role": "<|User|>",
                "content": prompt,  # Store original prompt
                "images": images,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # For the model input, we need to add the image placeholder
            model_turn = {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt_text}",
                "images": images,
            }

            if chat_mode:
                # Add chat history context (last 3 turns)
                conversation_context = []
                if len(self.chat_history) > 0:
                    for hist in self.chat_history[-3:]:
                        conversation_context.append(hist)

                # Add current turn to context
                conversation_context.append(model_turn)
                conversation_context.append({"role": "<|Assistant|>", "content": ""})
            else:
                # Single turn mode
                conversation_context = [
                    model_turn,
                    {"role": "<|Assistant|>", "content": ""},
                ]

            # Process input
            try:
                inputs = processor(
                    conversations=conversation_context,
                    images=images,
                    force_batchify=True
                ).to(device)
            except RuntimeError as e:
                inputs = processor(
                    conversations=conversation_context,
                    images=images
                ).to(device)

            # Generate response
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            generation_config = ModelManager.get_generation_config(
                processor, inputs_embeds, inputs.attention_mask,
                max_tokens, temperature, top_p
            )
            
            with torch.inference_mode():
                outputs = model.language_model.generate(**generation_config)
            
            response = processor.tokenizer.decode(
                outputs[0].cpu().tolist(),
                skip_special_tokens=True
            ).strip()
            
            if chat_mode:
                # Update chat history
                self.chat_history.append(current_turn)  # Add user turn with original prompt
                assistant_turn = {
                    "role": "<|Assistant|>",
                    "content": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.chat_history.append(assistant_turn)
                
                # Keep last 10 rounds
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
                
                formatted_history = ChatManager.format_chat_history(self.chat_history)
                return (response, formatted_history)
            else:
                # Single turn mode
                return (response, "")

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            print(error_msg)
            return (error_msg, "")

        @classmethod
        def IS_CHANGED(cls, **kwargs):
            return float("nan")

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VisionModelLoader": VisionModelLoader,
    "UnifiedVisionAnalyzer": UnifiedVisionAnalyzer
}

# Display Names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "VisionModelLoader": "Janus-7b-Pro Model Loader (Upload)",
    "UnifiedVisionAnalyzer": "Janus Vision 7b Pro (Chat)"
}