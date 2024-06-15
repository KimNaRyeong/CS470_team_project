import h5py
import numpy as np
import os
from PIL import Image
import scipy.io
import tqdm

if __name__ == '__main__':
    # jpg is RGB, png is depth

    if not os.path.isdir("../dataset/nyu_depth_v2/h5"):
        os.mkdir("../dataset/nyu_depth_v2/h5")
    
    current_directory = os.getcwd().replace("depth_cov", "")
    #print(current_directory)

    for root, dirs, files in tqdm.tqdm(os.walk("../dataset/nyu_depth_v2/official_splits/train/")):
        #print(root, dirs, files)
        #exit(0)
        label = root.split("/")[-1]

        for file in files:
            
            file_names = file.split(".")

            if file_names[-1] != "jpg":
                continue
            
            file_id = file_names[0].split("_")

            rgb_path = root + "/" + "rgb_" + file_id[-1] + ".jpg"
            depth_path = root + "/" + "sync_depth_" + file_id[-1] + ".png"

            rgb_image = np.array(Image.open(rgb_path))

            #print(rgb_image.shape)
            depth_image = np.array(Image.open(depth_path))

            rgb_image = np.transpose(rgb_image, (2, 1, 0))
            depth_image = np.expand_dims(depth_image, axis=2)
            depth_image = np.transpose(depth_image, (2, 1, 0))

            # print(rgb_image.shape)
            # print(depth_image.shape)
            # exit(0)
            with h5py.File(f"../dataset/nyu_depth_v2/h5/{file_id[-1]}.h5", "w") as f:
                f.create_dataset("rgb", data=rgb_image)
                f.create_dataset("depth", data=depth_image)

            with open("../dataset/nyu_depth_v2/nyudepthv2_train.txt", "a") as ff:
                ff.write(f"{current_directory}dataset/nyu_depth_v2/h5/{file_id[-1]}.h5\n")
    
    if not os.path.isdir("../dataset/nyu_depth_v2/h5_test"):
        os.mkdir("../dataset/nyu_depth_v2/h5_test")
    

    for root, dirs, files in tqdm.tqdm(os.walk("../dataset/nyu_depth_v2/official_splits/test/")):

        label = root.split("/")[-1]

        for file in files:
            
            file_names = file.split(".")

            if file_names[-1] != "jpg":
                continue
            
            file_id = file_names[0].split("_")

            rgb_path = root + "/" + "rgb_" + file_id[-1] + ".jpg"
            depth_path = root + "/" + "sync_depth_" + file_id[-1] + ".png"

            rgb_image = np.array(Image.open(rgb_path))
            depth_image = np.array(Image.open(depth_path))
            
            rgb_image = np.transpose(rgb_image, (2, 1, 0))
            depth_image = np.expand_dims(depth_image, axis=2)
            depth_image = np.transpose(depth_image, (2, 1, 0))
            
            with h5py.File(f"../dataset/nyu_depth_v2/h5_test/{file_id[-1]}.h5", "w") as f:
                f.create_dataset("rgb", data=rgb_image)
                f.create_dataset("depth", data=depth_image)

            with open("../dataset/nyu_depth_v2/nyudepthv2_val.txt", "a") as ff:
                ff.write(f"{current_directory}dataset/nyu_depth_v2/h5_test/{file_id[-1]}.h5\n")

    # with h5py.File('../dataset/nyu2/nyu_depth_v2_labeled.mat', 'r') as f:
    #     print(f["sceneTypes"])
            

            
            