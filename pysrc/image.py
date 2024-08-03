from PIL import Image
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

def image_to_tensor(image: ndarray, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by PIL.Image, the data range is [0, 255] or [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch
    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor

def tensor_to_image(tensor: Tensor, half: bool) -> Image:
    """Convert the Tensor(NCWH) data type supported by PyTorch to a PIL.Image image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (PIL.Image.Image): Image data supported by PIL
    """
    if half:
        tensor = tensor.half()

    # Convert tensor to numpy array
    image_array = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    return image

def load_image(image_path: str, half: bool, device: torch.device) -> Tensor:
    # read an image using PIL
    image = Image.open(image_path).convert('RGB')

    # Convert the image to a NumPy array
    image = np.array(image).astype(np.float32) / 255.0

    # Convert RGB image channel data to image formats supported by PyTorch
    tensor = image_to_tensor(image, half).unsqueeze_(0)

    # Data transfer to the specified device
    tensor = tensor.to(device, non_blocking=True)

    return tensor
