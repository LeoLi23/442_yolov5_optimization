import os
import shutil
import random
from PIL import Image

# TODO: Consider finding all image widths and heights (once per directory) instead of for EVERY image to speed things up
# NOTE: Estimated time to complete image copying for the entire thing is 60-70 minutes - Should be executed BEFORE a meeting
# And be thoroughly checked that label files are correct.

# Recommendation: Test training on a small subset of classes and check performance on that before expanding to ALL 21 classes.

classes = [  
  'ArmyDiver',
  'Ballena',
  'BlueFish',
  'BoySwimming',
  'CenoteAngelita',
  'DeepSeaFish',
  'Dolphin',
  'Fisherman',
  'FishFollowing',
  'GarryFish',
  'HoverFish',
  'JerkbaitBites',
  'MonsterCreature',
  'Octopus',
  'PinkFish',
  'SeaDiver',
  'SeaDragon',
  'SeaTurtle',
  'Steinlager',
  'WhaleAtBeach',
  'WhaleDiving',
  'WhiteShark']
def match_class(class_name):
    for i in range(len(classes)):
         if classes[i] in class_name:
              return i
    print(f"{class_name} not found in classes list") # This is bad

def create_dataset(root_dir, output_dir, split_ratio=[0.6, 0.2, 0.2]):
    """
    Creates a dataset from the given directory structure.

    Parameters:
    root_dir (str): The path to the root directory of the dataset.
    output_dir (str): The path to the output directory where the dataset will be created.
    split_ratio (list): A list of three float values that specifies the train/val/test split. The default is [0.7, 0.2, 0.1].

    Returns:
    None
    """
    assert sum(split_ratio) == 1.0, "Split ratio should add up to 1.0"
    assert len(split_ratio) == 3, "Split ratio should be a list of three float values"

    # Create output directory structure
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    # Get all image files and their corresponding label files
    image_files = []
    label_files = []
    for root, dirs, files in os.walk(root_dir):
        if 'img' in dirs:
            img_dir = os.path.join(root, 'img')
            label_file = os.path.join(root, 'groundtruth_rect.txt')
            with open(label_file, 'r') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        image_files.append(os.path.join(img_dir, f'{i+1}.jpg'))
                        label_files.append(line)

    # Split image and label files into train/val/test sets
    num_files = len(image_files)
    indices = list(range(num_files))
    random.Random(50).shuffle(indices)
    train_split = int(num_files * split_ratio[0])
    val_split = int(num_files * (split_ratio[0] + split_ratio[1]))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # Copy image files to output directory and  write label files to output directory
    for i, idx in enumerate(train_indices):     
        # Get width and height of image - resize to one dimension as 640
        with Image.open(image_files[idx]) as img:
                    width, height = img.size
                    if width >= height:
                         new_width = 640
                         new_height = int(round(640 * (float(height)/width)))
                    else:
                        new_width = int(round(640 * (float(width)/height)))
                        new_height = 640
                    img = img.resize((new_width, new_height))
                    # Save resized img
                    img.save(os.path.join(output_dir, 'images', 'train', f'{i}.jpg'))
                    #shutil.copyfile(image_files[idx], os.path.join(output_dir, 'images', 'train', f'{i}.jpg'))

        with open(os.path.join(output_dir, 'labels', 'train', f'{i}.txt'), 'w') as f:
            # Parse data from input
            top_leftx, top_lefty, box_width, box_height = [float(x) for x in label_files[idx].split()]

            x_center = top_leftx + box_width / 2.0
            y_center = top_lefty + box_height / 2.0

            obj_class = match_class(image_files[idx].split('/')[-3])
            # Write normalized data to file
            f.write(f"{obj_class} {x_center / width} {y_center / height} {box_width / width} {box_height / height}")
            # if obj_class == None or x_center > width or y_center > height or box_width > width or box_height > height or x_center < 0 or y_center < 0 or box_width < 0 or box_height < 0:
            #     print(f"BAD!: {obj_class} {x_center / width} {y_center / height} {box_width / width} {box_height / height}")
    for i, idx in enumerate(val_indices):
        # Get width and height of image - resize to one dimension as 640
        with Image.open(image_files[idx]) as img:
                    width, height = img.size
                    if width >= height:
                         new_width = 640
                         new_height = int(640 * (float(height)/width))
                    else:
                        new_width = int(640 * (float(width)/height))
                        new_height = 640
                    img = img.resize((new_width, new_height))
                    # Save resized img
                    img.save(os.path.join(output_dir, 'images', 'val', f'{i}.jpg'))
                    #shutil.copyfile(image_files[idx], os.path.join(output_dir, 'images', 'train', f'{i}.jpg'))

        with open(os.path.join(output_dir, 'labels', 'val', f'{i}.txt'), 'w') as f:
            # Parse data from input
            top_leftx, top_lefty, box_width, box_height = [float(x) for x in label_files[idx].split()]

            x_center = top_leftx + box_width / 2.0
            y_center = top_lefty + box_height / 2.0

            obj_class = match_class(image_files[idx].split('/')[-3])
            # Write normalized data to file
            f.write(f"{obj_class} {x_center / width} {y_center / height} {box_width / width} {box_height / height}")
    for i, idx in enumerate(test_indices):
        # Get width and height of image - resize to one dimension as 640
        with Image.open(image_files[idx]) as img:
                    width, height = img.size
                    if width >= height:
                         new_width = 640
                         new_height = int(640 * (float(height)/width))
                    else:
                        new_width = int(640 * (float(width)/height))
                        new_height = 640
                    img = img.resize((new_width, new_height))
                    # Save resized img
                    img.save(os.path.join(output_dir, 'images', 'test', f'{i}.jpg'))
                    #shutil.copyfile(image_files[idx], os.path.join(output_dir, 'images', 'train', f'{i}.jpg'))
        with open(os.path.join(output_dir, 'labels', 'test', f'{i}.txt'), 'w') as f:
            # Parse data from input
            top_leftx, top_lefty, box_width, box_height = [float(x) for x in label_files[idx].split()]

            x_center = top_leftx + box_width / 2.0
            y_center = top_lefty + box_height / 2.0

            obj_class = match_class(image_files[idx].split('/')[-3])
            # Write normalized data to file
            f.write(f"{obj_class} {x_center / width} {y_center / height} {box_width / width} {box_height / height}")

create_dataset('./dataset/UOT_raw/', './dataset/UOT_dataset')