import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 as cv


defect_labels = ['dent', 'Particle', 'scratch', 'smudge','background']

class SurfaceDefectDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512))])
        
        img_files = os.listdir(root_dir)
        self.defect_types = []
        self.images = []
        for file_name in img_files:
            # print(file_name)
            defect_index = defect_labels.index(file_name)
            
            for file in os.listdir(os.path.join(root_dir,file_name)):
                self.images.append(os.path.join(root_dir,file_name,file))
                self.defect_types.append(defect_index)
                # print(os.path.join(root_dir,file_name,file),"label------->",defect_index)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image_path = self.images[idx] # 通过索引获取图片路径和图片名
        img = cv.imread(image_path) # BGR
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        sample = {'image': self.transform(img), 'label': self.defect_types[idx]}
        return sample

# class SurfaceDefectDataset(Dataset):
#     def __init__(self, root_dir):
#         self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#                 transforms.Resize((512, 512))])
        
#         img_files = os.listdir(root_dir)
#         self.defect_types = []
#         self.images = []
#         for file_name in img_files:
#             defect_class = file_name.split('_')[0]  # 以下划线分割文件名
#             defect_index = defect_labels.index(defect_class) # 缺陷类别转为索引
#             self.images.append(os.path.join(root_dir, file_name)) # 图片路径和图片名
#             self.defect_types.append(defect_index) # 缺陷索引
#     def __len__(self):
#         return len(self.images)
#     def __getitem__(self, idx):
#         image_path = self.images[idx] # 通过索引获取图片路径和图片名
#         img = cv.imread(image_path) # BGR
#         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#         sample = {'image': self.transform(img), 'label': self.defect_types[idx]}
#         return sample

if __name__ == '__main__':
    ds = SurfaceDefectDataset("E:\\nano\\1_projects\\code\\develop\\classfy\\data\\mobilenet\\data1\\train")
    # print(len(ds))
    # print(ds[0]['image'].shape, ds[0]['label'])

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    sample = next(iter(dl))
    print(type(sample))
    print(sample['image'].shape)

    # img=cv.imread("E:\\nano\\1_projects\\code\\develop\\classfy\\data\\cnn_data\\train\\dent_1.jpeg")
    # print(img.shape)