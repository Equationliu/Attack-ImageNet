import pandas as pd
import numpy as np
from PIL import Image 
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, csv_path, img_path = "./../images/"):
        # Preprocess
        self.to_tensor = transforms.ToTensor()

        self.data_info = pd.read_csv(csv_path, header=0)
        self.image_name = np.asarray(self.data_info.iloc[:, 0])
        self.label = np.asarray(self.data_info.iloc[:, 1]) - 1
        self.target_label = np.asarray(self.data_info.iloc[:, 2]) - 1
        self.data_len = len(self.data_info.index)
        self.img_path = img_path
        
    def __getitem__(self, index):
        single_image_name = self.image_name[index]
        img_as_img = Image.open(self.img_path + single_image_name)

        img_as_tensor = self.to_tensor(img_as_img)
        
        single_image_label = self.label[index]
        single_image_target_label = self.target_label[index]
        single_image_name = self.image_name[index]

        return (img_as_tensor, single_image_label, single_image_target_label, single_image_name)

    def __len__(self):
        return self.data_len