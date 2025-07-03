import torch
import torchvision
import torch.backends.cudnn as cudnn
import os
from osgeo import gdal
from torchvision.transforms import functional as F
import argparse
import cv2
from indexs import psnr,ergas,sam,ssim,qindex
import numpy as np
from scipy.interpolate import Rbf




class Tiff(object):
    def readTif_to_Ndarray(self, img_path):
        dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
        # 获得矩阵的列数
        self.width = dataset.RasterXSize
        # 栅格矩阵的行数
        self.height = dataset.RasterYSize

        # 获得数据
        self.data = dataset.ReadAsArray(0, 0, self.width, self.height)
        C,W,H = self.data.shape
        return np.reshape(self.data,((H, W, C)))
        '''
		这里的permute函数是变换维度的函数，我输入的tif维度是[7,448,448]，但是
		gdal读取的时候是[448,7,448]和cv2的读取方式相似，所以这里用permute(1,0,2)把
		维度转换为正常的[C,W,H]
		'''
    def Ndarray_to_Tensor(self, ndarray):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        s = len(ndarray.shape)
        if s == 2:
            t = F.to_tensor(ndarray).unsqueeze(0)

            return t.to(device)
        else:
            

            t = F.to_tensor(ndarray).permute(1, 0, 2).unsqueeze(0)

        return t.to(device)

    def Tensor_to_Ndarray(self,tensor):
        #t = F.to_tensor(ndarray).permute(1, 0, 2)
        tensor = tensor.permute(0,1,3,2)
        t = tensor.cpu().numpy()
        t = t.squeeze(0)
        return t

        #  保存tif文件函数
    def writeTiff(self, im_data, width, height, bands, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
    # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(width), int(height), int(bands), datatype)
        for i in range(bands):    #multi-bands
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset

    def Save(self,input_path,array_img,output_path,new_name):
        dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
        # 获得矩阵的列数
        width = dataset.RasterXSize
        # 栅格矩阵的行数
        height = dataset.RasterYSize
        
        bands = dataset.RasterCount
        

        self.writeTiff(array_img, width, height, bands, output_path + new_name)

if __name__ == '__main__':

    hs_folder = r"C:\baidunetdiskdownload\Pan_sharpening\full_examples\reduced_test_qb_gt"
    preds_folder = r"C:\baidunetdiskdownload\NEW\qb"

    total_psnr_low = 0
    total_sam_low = 0
    total_ergas_low = 0
    total_ssim_low = 0
    total_psnr = 0
    total_sam = 0
    total_ergas = 0
    total_ssim = 0
    total_q2n = 0

    rt=Tiff()
    for i in range (0,20):

        real_img=rt.readTif_to_Ndarray(hs_folder+ "/%d.tif" % i)  #Shape: (bands, height, width)
        print(real_img.shape)
        #real_img=rt.Ndarray_to_Tensor(real_img)
       
        #preds = rt.readTif_to_Ndarray(preds_folder+ "/%d.tif" % i)  #Shape: (bands, height, width)
        preds=rt.readTif_to_Ndarray((preds_folder+"/{0}_{1}.tif".format('NEW_PAN','%d'%i)))
        # M,N,C = preds.shape 
        # preds = np.reshape(preds,(C,M,N))
        #preds=rt.Ndarray_to_Tensor(preds)

        sam_score = sam(real_img,preds)
        psnr_score = psnr(real_img, preds)  
        ssim_score = ssim(real_img, preds) 
        ergas_score = ergas(real_img,preds)
        qindex_score = qindex(real_img,preds)
        total_psnr = total_psnr+psnr_score
        total_ssim = total_ssim+ssim_score
        total_sam = total_sam+sam_score
        total_ergas = total_ergas+ergas_score
        total_q2n = total_q2n + qindex_score

        
        # #low refence
        # psnr_low = calc_mean_psnr(real_img, label_img)   
        # ssim_low = calc_ssim(real_img,label_img)
        # sam_low,ergas_low = compute_index(real_img,label_img)
        # total_psnr_low = total_psnr_low+psnr_low
        # total_ssim_low = total_ssim_low+ssim_low
        # total_sam_low = total_sam_low+sam_low
        # total_ergas_low = total_ergas_low+ergas_low


    
    print('PSNR: {:.2f}'.format(total_psnr/20.0))  # 格式化输出PSNR
    print('SSIM: {:.2f}'.format(total_ssim/20.0))  
    print('SAM: {:.2f}'.format(total_sam/20.0))
    print('ERGAS:{:.2f}'.format(total_ergas/20.0))
    print('Q2n:{:.2f}'.format(total_q2n/20.0))

    # print('low PSNR: {:.2f}'.format(total_psnr_low/20.0))  # 格式化输出PSNR
    # print('low SSIM: {:.2f}'.format(total_ssim_low/20.0))  # 格式化输出PSNR
    # print('low SAM: {:.2f}'.format(total_sam_low/20.0))
    # print('low ERGAS:{:.2f}'.format(total_ergas_low/20.0))

    


    