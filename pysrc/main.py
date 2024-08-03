# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import model
from image import load_image, tensor_to_image
from util import load_pretrained_state_dict

def infer(input_path, output_path, model_path, device, half_precision):
    device = torch.device(device)
    input_tensor = load_image(input_path, half_precision, device)

    sr_model = model.RRDBNet().to(device)
    sr_model = load_pretrained_state_dict(sr_model, model_path)

    # Start the verification mode of the model.
    sr_model.eval()

    # Enable half-precision inference to reduce memory usage and inference time
    if half_precision:
        sr_model.half()

    with torch.no_grad():
        sr_tensor = sr_model(input_tensor)

    # Convert back to image and save
    cr_image = tensor_to_image(sr_tensor, half_precision)
    cr_image.save(output_path)