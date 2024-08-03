# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
# =============================================================================

import torch
from torch import nn

def load_state_dict(
        model: nn.Module,
        state_dict: dict,
):
    """Load model weights and parameters

    Args:
        model (nn.Module): model
        state_dict (dict): model weights and parameters waiting to be loaded

    Returns:
        model (nn.Module): model after loading weights and parameters
    """
    model_state_dict = model.state_dict()

    # Filter parameters to include only those that match size and name
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict and v.size() == model_state_dict[k].size()}

    # Update and load the model state dictionary
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_pretrained_state_dict(
        model: nn.Module,
        model_weights_path: str,
) -> nn.Module:
    """Load pre-trained model weights

    Args:
        model (nn.Module): model
        model_weights_path (str): model weights path

    Returns:
        model (nn.Module): the model after loading the pre-trained model weights
    """

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["state_dict"]
    model = load_state_dict(model, state_dict)
    return model