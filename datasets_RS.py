import torch
import torchvision
from torch.utils.data import Dataset
import os
from osgeo import gdal
from torchvision.transforms import functional as F


class ReadTiff(object):
    def readTif_to_Ndarray(self, img_path):
        dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
        # 获得矩阵的列数
        self.width = dataset.RasterXSize
        # 栅格矩阵的行数
        self.height = dataset.RasterYSize
        # 获得数据
        self.data = dataset.ReadAsArray(0, 0, self.width, self.height)
        return self.data
        '''
		这里的permute函数是变换维度的函数，我输入的tif维度是[7,448,448]，但是
		gdal读取的时候是[448,7,448]和cv2的读取方式相似，所以这里用permute(1,0,2)把
		维度转换为正常的[C,W,H]
		'''
    def Ndarray_to_Tensor(self, ndarray):
        s = len(ndarray.shape)
        if s == 2:
            t = F.to_tensor(ndarray)

            return t
        else:
            

            t = F.to_tensor(ndarray).permute(1, 0, 2)
        #t = F.to_tensor(ndarray)
        #t_1=t.unsqueeze(0)
            return t
 
class MyDataset(Dataset):#继承了Dataset子类
    def __init__(self,input_root,label_root):
        #分别读取输入/标签图片的路径信息
        self.input_root=input_root
        self.input_files=os.listdir(input_root)#列出指定路径下的所有文件
 
        self.label_root=label_root
        self.label_files=os.listdir(label_root)

    def __len__(self):
        #获取数据集大小
        return len(self.input_files)
    def __getitem__(self, index):
        #根据索引(id)读取对应的图片
        rt=ReadTiff()
        input_img_path=os.path.join(self.input_root,self.input_files[index])
        input_img=rt.readTif_to_Ndarray(input_img_path)
        input_img=rt.Ndarray_to_Tensor(input_img)
 
        label_img_path=os.path.join(self.label_root,self.label_files[index])
        label_img=rt.readTif_to_Ndarray(label_img_path)
        label_img=rt.Ndarray_to_Tensor(label_img)
        s = label_img.shape
        
        return (input_img,label_img)#返回成对的数据

class MyDataset_Pan(Dataset):#继承了Dataset子类
    def __init__(self,pan_root,input_root,label_root):
        #分别读取输入/标签图片的路径信息
        self.pan_root=pan_root
        self.pan_files=os.listdir(pan_root)#列出指定路径下的所有文件

        self.input_root=input_root
        self.input_files=os.listdir(input_root)#列出指定路径下的所有文件
 
        self.label_root=label_root
        self.label_files=os.listdir(label_root)

    def __len__(self):
        #获取数据集大小
        return len(self.input_files)
    def __getitem__(self, index):
        #根据索引(id)读取对应的图片
        rt=ReadTiff()
        pan_img_path=os.path.join(self.pan_root,self.pan_files[index])
        pan_img=rt.readTif_to_Ndarray(pan_img_path)
        pan_img=rt.Ndarray_to_Tensor(pan_img)

        input_img_path=os.path.join(self.input_root,self.input_files[index])
        input_img=rt.readTif_to_Ndarray(input_img_path)
        input_img=rt.Ndarray_to_Tensor(input_img)
 
        label_img_path=os.path.join(self.label_root,self.label_files[index])
        label_img=rt.readTif_to_Ndarray(label_img_path)
        label_img=rt.Ndarray_to_Tensor(label_img)
        s = label_img.shape
        
        return (pan_img,input_img,label_img)#返回成对的数据

# file = r'D:\BaiduNetdiskDownload\training_wv3\gt\0.tif'
# t = ReadTiff()
# dataset = t.readTif_to_Ndarray(file)
# print(dataset.shape)
# torch_dataset = t.Ndarray_to_Tensor(dataset)
# print(torch_dataset.shape)