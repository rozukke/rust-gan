from PIL import Image
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

def image_to_tensor(image: ndarray, half: bool) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Use half precision in model inference to trade accuracy for speed
    if half:
        tensor = tensor.half()

    return tensor

def tensor_to_image(tensor: Tensor, half: bool) -> Image:
    if half:
        tensor = tensor.half()
    
    # Convert to tensor within required bounds
    image_array = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    image = Image.fromarray(image_array)
    
    return image

def load_image(image_path: str, half: bool, device: torch.device) -> Tensor:
    image = Image.open(image_path).convert('RGB')

    image = np.array(image).astype(np.float32) / 255.0

    # Convert RGB image channel data to image format supported by PyTorch
    tensor = image_to_tensor(image, half).unsqueeze_(0).to(device, non_blocking=True)

    return tensor
