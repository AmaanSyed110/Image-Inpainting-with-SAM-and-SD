# Image-Inpainting-with-SAM-and-SD

## Overview
This project demonstrates advanced image inpainting using the Segment Anything Model (SAM) and Stable Diffusion to deliver state-of-the-art results. SAM is employed to precisely identify regions of interest in the image, providing an accurate mask for the inpainting process. Once these regions are detected, Stable Diffusion is utilized to seamlessly replace the identified areas with new content generated based on a given text prompt, ensuring the inpainted content aligns with the user’s intent. The pipeline is designed to efficiently integrate these cutting-edge machine learning models, leveraging their strengths to produce high-quality and context-aware results. By utilizing GPU acceleration, the process achieves remarkable speed and performance, making it suitable for real-world applications such as creative content generation, image restoration, and object removal.

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

## Contributions
Contributions are welcome! Please fork the repository and create a pull request for any feature enhancements or bug fixes.

