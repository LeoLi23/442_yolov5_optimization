import os
import shutil

def get_all_image_files_in_folder(src_folder):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    img_files = []
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            if any(file_name.endswith(ext) for ext in img_extensions):
                img_files.append(os.path.join(root, file_name))
    return img_files

def copy_images_to_destination(src_img_files, dest_folder):
    for src_img_path in src_img_files:
        img_name = os.path.basename(src_img_path)
        img_folder_parent = os.path.basename(os.path.dirname(os.path.dirname(src_img_path)))
        dest_img_name = f"{img_folder_parent}_{img_name}"
        dest_img_path = os.path.join(dest_folder, dest_img_name)
        shutil.copy2(src_img_path, dest_img_path)

# images/1.png
# labels/1.txt

# x_center: (x_max + x_min)/img_width
# y_center: (y_max + y_min)/img_height
# width: (x_max - x_min)/img_width
# height: (y_max - y_min)/img_height

def convert_annotation_yolo(ground_truth_file):
    ground_truth = open(ground_truth_file, "r")

    return

base_folder = '../dataset/UOT_dataset'
dest_folder = '../dataset/UOT_dataset/images'

# Ensure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

for folder_name in os.listdir(base_folder):
    src_folder = os.path.join(base_folder, folder_name)

    # Skip the destination folder and non-directory items
    if folder_name == 'images' or not os.path.isdir(src_folder):
        continue

    # Get all image files in the source folder
    #src_img_files = get_all_image_files_in_folder(src_folder)

    # Copy images to the destination folder
    #copy_images_to_destination(src_img_files, dest_folder)
