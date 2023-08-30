from dataclasses import dataclass
from typing import Union, List

import PIL
import numpy as np
from diffusers import ImagePipelineOutput


@dataclass
class DBADPipelineOutput(ImagePipelineOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        history:
            List of produced images for each timestep.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    history: List