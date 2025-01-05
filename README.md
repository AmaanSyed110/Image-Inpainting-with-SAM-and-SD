# Image-Inpainting-with-SAM-and-SD

## Overview
This project demonstrates advanced image inpainting using the Segment Anything Model (SAM) to identify regions of interest and Stable Diffusion to replace those regions with new image content based on a text prompt. The pipeline efficiently integrates machine learning models to achieve high-quality inpainted images while leveraging GPU acceleration.

## Key Features
- **Image Segmentation**: Utilizes SAM to create precise masks for regions of interest.

- **Text-Prompted Inpainting**: Employs Stable Diffusion to replace masked areas with content generated from user-defined text prompts.

- **Interactive Workflow**: Users can define regions using specific points on the image.

- **High-Quality Output**: Produces seamless and high-resolution inpainted images.

## Tech Stack
- **torch**: For machine learning computations and GPU acceleration.

- **torchvision**: Image manipulation utilities.

- **numpy**: Numerical computations.

- **cv2**: OpenCV for image processing.

- **matplotlib.pyplot**: Visualization of images and results.

- **sam_model_registry, SamPredictor**: Segment Anything Model components for creating masks.

- **StableDiffusionInpaintPipeline**: Inpainting functionality using Stable Diffusion.

## Example Workflow

```python
# Import necessary libraries
import cv2
import torch
from sam_model_registry import SamPredictor
from stable_diffusion import StableDiffusionInpaintPipeline

# Load the image
image = cv2.imread('input.jpg')

# Initialize SAM for mask creation
sam = SamPredictor()
mask = sam.predict(image, point=(x, y))

# Use Stable Diffusion for inpainting
pipeline = StableDiffusionInpaintPipeline.from_pretrained('model-path')
inpainted_image = pipeline(image, mask, prompt="A sunny beach")

# Display the output
plt.imshow(inpainted_image)
plt.show()
```

