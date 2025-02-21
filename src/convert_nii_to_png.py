import os
import glob
import shutil
import numpy as np
import nibabel as nib
import imageio

def convert_nii_to_png(input_dir, output_dir, rotate_ct=False):
    """
    Convert .nii files to .png format.

    Args:
        input_dir (str): Directory containing .nii files.
        output_dir (str): Directory to save .png files.
        rotate_ct (bool): Whether to rotate CT scans 90 degrees (default: False).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for nii_file in glob.glob(os.path.join(input_dir, "*.nii")):
        try:
            image_array = nib.load(nii_file).get_fdata()
            print(f"Processing {nii_file}: Shape {image_array.shape}, Type {image_array.dtype}")
            data = np.rot90(image_array[:, :, 0]) if rotate_ct else image_array[:, :, 0]
            image_name = os.path.basename(nii_file)[:-4] + ".png"
            output_path = os.path.join(output_dir, image_name)
            imageio.imsave(output_path, data.astype(np.float32))
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"Error processing {nii_file}: {e}")

    print(f"Finished converting images from {input_dir} to {output_dir}")
