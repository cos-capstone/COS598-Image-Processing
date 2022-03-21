"""I could have used sitk.GetPixel() and sitk.SetPixel(), 
but I decided against it because I wanted to explore more 
into numpy's ndarrays since I have been needing to learn 
them for other things recently.

Also - it seems like sitk's GetImageFromArray implicitly 
converts every pixel value to Float64, so I could not figure
out a way to save the image besides going into ImageJ and 
saving them manually.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import SimpleITK as sitk


def _default_image():
    return sitk.Image(256, 256, sitk.sitkUInt8)


def gaussian_transform(
    μ: tuple[float, float],
    σ: tuple[float, float],
    θ: float,
    a: int = 1,
    b: Optional[np.ndarray] = sitk.GetArrayFromImage(_default_image()),
):
    """The main function. Applies a gaussian transformation to a given `b`
    image with constraints μ, σ, θ, and a, respectively. Given images are 
    generated at the bottom of this file.
    """
    μ = np.asarray(μ)
    σ = np.asarray(σ)

    R = np.array([[np.cos(θ), np.sin(θ)], [-np.sin(θ), np.cos(θ)]])

    Σ = np.array([[σ[0] ** 2, 0], [0, σ[1] ** 2]])
    Σ_inv = np.linalg.inv(Σ)
    
    # Non-destructive: create a new ndarray
    res = np.zeros(b.shape)
    for i, x in enumerate(b):
        for j, y in enumerate(x):
            mag = np.array([i, j]) - μ
            
            # Defined function `f`
            transient_res = mag @ R @ Σ_inv @ np.transpose(R) @ np.transpose(mag)

            transient_res = np.exp((-1 / 2) * transient_res)

            val = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Σ)))
            val = val * transient_res
            val = a * val + y

            # Set the pixel to the computed value
            res[i][j] = val
    return res

image_viewer = sitk.ImageViewer()

# Image 0 - Test

# image = sitk.Image(256, 256, sitk.sitkUInt32)
# array = sitk.GetArrayFromImage(image)

# # Convert the image
# array2 = gaussian_transform(μ=(128, 128), σ=(45, 5), θ=0, a=150000, b=array)

# # Get the image from the converted matrix
# image2 = sitk.GetImageFromArray(array2)

# Image 1

converted_array_1 = gaussian_transform(μ=(128, 128), σ=(20, 20), θ=0, a=100000)

converted_image_1 = sitk.GetImageFromArray(converted_array_1)
# sitk.WriteImage(converted_image_1, "image_1.png")
image_viewer.Execute(converted_image_1)

# Image 2

converted_array_2 = gaussian_transform(μ=(128, 128), σ=(5, 20), θ=np.pi/4, a=100000)

converted_image_2 = sitk.GetImageFromArray(converted_array_2)
# sitk.WriteImage(converted_image_2, "image_2.png")
image_viewer.Execute(converted_image_2)

# Image 3

converted_array_3 = gaussian_transform(μ=(128, 128), σ=(10, 30), θ=(-np.pi/6), a=100000)

converted_image_3 = sitk.GetImageFromArray(converted_array_3)
# sitk.WriteImage(converted_image_3, "image_3.jpeg")
image_viewer.Execute(converted_image_3)