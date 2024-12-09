import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the NIfTI file
file_path = '/Users/jasonjoelpinto/Downloads/ADNI 2/133_S_1031/MPR-R__GradWarp__N3__Scaled/2006-11-16_09_31_08.0/I40012/ADNI_133_S_1031_MR_MPR-R__GradWarp__N3__Scaled_Br_20070213232512977_S22621_I40012.nii'
img = nib.load(file_path)

# Get the image data as a numpy array
data = img.get_fdata()

# Check the dimensions of the data (e.g., 256x256x150)
print(data.shape)

# Save slices as PNG (e.g., middle slice of the third dimension)
output_folder = "/Users/jasonjoelpinto/Downloads/ADNI 2/133_S_1031/MPR-R__GradWarp__N3__Scaled/2006-11-16_09_31_08.0/I40012/"
slice_index = data.shape[2] // 2  # Choose a slice
slice_data = data[:, :, slice_index]

# Normalize to 0-255 for saving as image
slice_data = ((slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255).astype(np.uint8)

# Save the slice
output_file = f"{output_folder}/slice_{slice_index}.png"
Image.fromarray(slice_data).save(output_file)

print(f"Saved: {output_file}")