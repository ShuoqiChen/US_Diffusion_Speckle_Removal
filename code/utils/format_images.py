import os, sys
from PIL import Image
import os
import glob
import torch
import numpy as np

def rename_images(image_dir):
    # List all files in the directory
    files = os.listdir(image_dir)
    
    # Filter out only image files (you can add more extensions if needed)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Sort the files to ensure consistent ordering
    image_files.sort()
    
    # Rename each image file
    for index, filename in enumerate(image_files):
        # Get the file extension
        file_extension = os.path.splitext(filename)[1]
        
        # Create the new filename
        new_filename = f"img_{index}{file_extension}"
        
        # Construct full file paths
        old_file = os.path.join(image_dir, filename)
        new_file = os.path.join(image_dir, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")

    return

def resize_images(folder_path, width, height):
    # Get a list of all image files in the folder (you can add more extensions if needed)
    image_files = glob.glob(os.path.join(folder_path, '*.[jp][pn]g')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg')) + \
                  glob.glob(os.path.join(folder_path, '*.gif')) + \
                  glob.glob(os.path.join(folder_path, '*.bmp'))

    # Resize each image
    for file_path in image_files:
        with Image.open(file_path) as img:
            # Resize the image
            resized_img = img.resize((width, height))
            # Save the resized image back to the same path
            resized_img.save(file_path)

    print(f"Resized {len(image_files)} images to {width}x{height}.")

    return
        
def image_to_grayscale(input_folder, output_folder):
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Construct full file path
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpeg')

            # Open the PNG image
            image = Image.open(input_path)

            # Convert the image to grayscale
            grayscale_image = image.convert('L')

            # Save the grayscale image as a JPEG
            grayscale_image.save(output_path, 'JPEG')

            print(f"Converted {input_path} to {output_path}")
    
def image_to_tensor(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)
    
    # Resize the image to 64x64 if it's not already
    image = image.resize((64, 64))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Convert the numpy array to a torch tensor and add batch and channel dimensions
    tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    print(f"converted image tensor is {image_to_tensor}")

    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.imshow(tensor[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(os.path.join("/isi/git/US_Diffusion_Speckle_Removal/output/eval", "replace_image.jpg"))

    return tensor


if __name__ == "__main__":
    print("TODO:")
    # rename_images("/isi/git/US_Diffusion_Speckle_Removal/data/BreastUS")
    # resize_images('/isi/git/US_Diffusion_Speckle_Removal/data/BreastUS', width=64, height=64)
    image_to_grayscale("/isi/git/US_Diffusion_Speckle_Removal/data/BreastUS",
                                        "/isi/git/US_Diffusion_Speckle_Removal/data/BreastUS_copy")