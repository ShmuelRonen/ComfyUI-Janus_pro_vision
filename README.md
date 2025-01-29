# ComfyUI Janus Pro Vision

A ComfyUI custom node extension that integrates the Janus-Pro-7B vision-language model from DeepSeek AI on your's local computer, enabling powerful image understanding and multi-turn conversation capabilities.

#### Vision Mode (One or two images)
![image](https://github.com/user-attachments/assets/9f4b0575-2c6d-4c99-beca-2beaa41ef119)

#### Chat Mode (One or two images)
![Screenshot 2025-01-29 213437](https://github.com/user-attachments/assets/0fbe7876-b7d8-4124-966a-dbad249e0420)


## Features

- 🖼️ **Advanced Image Analysis**: Leverages Janus-Pro-7B's capabilities for detailed image understanding and description
- 💬 **Multi-turn Chat**: Supports interactive conversations about images with context awareness
- 🔄 **Dual Image Support**: Can analyze relationships between two images simultaneously
- 🚀 **Automatic Model Download**: Downloads model files automatically on first use
- ⚙️ **Flexible Configuration**: Customizable parameters for generation and image processing
- 🎯 **ComfyUI Integration**: Seamless integration with ComfyUI workflow

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-Janus_pro_vision.git
```

2. Install required dependencies:
```bash
pip install requests
pip install tqdm
```

The model files will be automatically downloaded on first use from DeepSeek's HuggingFace repository.

## Available Nodes

### 1. Janus-7b-Pro Model Loader (Upload)
Handles model loading and management.
- Input: None (uses default model path)
- Output: JANUS_MODEL (model object for use in analyzer)

### 2. Janus Vision 7b Pro (Chat)
Main analysis node with chat capabilities.

Inputs:
- `janus_model`: Model object from loader node
- `image_a`: Primary image for analysis
- `image_b`: (Optional) Secondary image for comparison
- `prompt`: Text prompt/question about the image(s)
- `chat_mode`: Enable/disable chat functionality
- `seed`: Random seed for generation
- `temperature`: Generation temperature (0.0 - 2.0)
- `top_p`: Top-p sampling parameter (0.0 - 1.0)
- `max_tokens`: Maximum generation length
- `image_size`: Target image size for processing (512-2048)
- `frame_size`: Border thickness for image display (1-10)
- `reset_chat`: Clear chat history

Outputs:
- `response`: Model's response text
- `chat_history`: Formatted chat history (in chat mode)

## Configuration

### Image Processing Parameters
- `image_size`: Controls the maximum dimension while maintaining aspect ratio (default: 1024)
    - Range: 512 to 2048 pixels
    - Steps: 64 pixels
    - Example: If image is 2000x1000px and image_size=1024:
        - Width will be scaled to 1024
        - Height will be scaled proportionally to 512

- `frame_size`: Border thickness for visual separation (default: 2)
    - Range: 1 to 10 pixels
    - Example values:
        - frame_size=1: Thin border
        - frame_size=2: Standard border
        - frame_size=5: Thick border
        - frame_size=10: Very thick border

### Generation Parameters
- `temperature`: Controls response randomness
    - 0.1: More focused and deterministic
    - 0.7: More creative and varied
- `top_p`: Nucleus sampling parameter (0.95 recommended)
- `max_tokens`: Maximum length of generated response

## Model Information

This extension uses the Janus-Pro-7B model from DeepSeek AI, which offers:
- Strong image understanding capabilities
- Multi-turn conversation support
- High-quality natural language generation
- Support for image comparison and analysis

## Requirements

- ComfyUI
- Python 3.8+
- PyTorch
- Transformers library
- requests
- tqdm

## License

This project is MIT licensed. The Janus-Pro-7B model has its own license from DeepSeek AI.

## Acknowledgments

- DeepSeek AI for the Janus-Pro-7B model
- ComfyUI community for the framework and support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
