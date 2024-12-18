import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_dementia_classes(directories):
    """
    Plot images from each dementia class in a clean, modern 2x2 grid.
    
    Args:
        directories (dict): Dictionary mapping class names to directory paths
    """
    # Create a modern, clean figure
    plt.figure(figsize=(16, 12), dpi=400, facecolor='#F5F5F5')
    plt.subplots_adjust(wspace=0.1, hspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Iterate through directories and plot images
    for i, (class_name, directory) in enumerate(directories.items()):
        # Get the first image in the directory
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # Create subplot in 2x2 grid
            plt.subplot(2, 2, i+1)
            
            # Open and process the image
            image_path = os.path.join(directory, image_files[0])
            img = Image.open(image_path)
            
            # Convert to grayscale and ensure consistent size
            img_array = np.array(img)
            
            # If color image, convert to grayscale
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=2)
            
            # Display image with tight cropping and aspect preservation
            plt.imshow(img_array, cmap='gray', aspect='equal')
            
            # Remove axes
            plt.gca().set_axis_off()
            
            # Add subtle border
            plt.gca().spines['top'].set_color('#3498db')
            plt.gca().spines['bottom'].set_color('#3498db')
            plt.gca().spines['left'].set_color('#3498db')
            plt.gca().spines['right'].set_color('#3498db')
            
            # Add class name closer to the image
            plt.text(0.5, -0.05, class_name, 
                     horizontalalignment='center', 
                     verticalalignment='top', 
                     transform=plt.gca().transAxes,
                     fontweight='bold',
                     fontsize=14)
    
    plt.tight_layout()
    plt.show()

# Directories for different dementia classes
dementia_dirs = {
    'Mild Demented': "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented",
    'Moderate Demented': "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented",
    'Non Demented': "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented",
    'Very Mild Demented': "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented"
}

# Call the visualization function
plot_dementia_classes(dementia_dirs)