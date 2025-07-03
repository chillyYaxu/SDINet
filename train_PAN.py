import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import copy
import time
import torchvision
import numpy as np
from torch import Tensor
import torch
from torch import nn
import torch.optim as optim
from datasets_RS import MyDataset_Pan
import losses
from PanNet import PanNet
from shufflemixer_losses import FFTLoss

# gpu加速库
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader

# 进度条
from tqdm import tqdm

from utils_SRCNN import AverageMeter, calc_psnr, calc_ssim, get_logger

##需要修改的参数
# epoch.pth
# losslog
# psnrlog
# best.pth

'''
python train.py --train-file "path_to_train_file" \
                --eval-file "path_to_eval_file" \
                --outputs-dir "path_to_outputs_file" \
                --scale 3 \
                --lr 1e-4 \
                --batch-size 16 \
                --num-epochs 400 \
                --num-workers 0 \
                --seed 123  
'''
if __name__ == '__main__':

    #对读取的图片采取的处理方法，详情自行搜索transforms的用法
    #transforms_imag=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # 初始参数设定
    parser = argparse.ArgumentParser()   # argparse是python用于解析命令行参数和选项的标准模块
    parser.add_argument('--base-file', type=str, default="")  # base path
    parser.add_argument('--train-file', type=str, default="")  # 训练 文件目录
    parser.add_argument('--train-pan', type=str, default="")  # 训练 文件目录
    parser.add_argument('--trainlabel-root', type=str, default="")  # 训练标签 文件目录
    parser.add_argument('--eval-file', type=str, default="")  # 测试 文件目录
    parser.add_argument('--eval-pan', type=str, default="")  # 测试 文件目录
    parser.add_argument('--evallabel-root', type=str, default="")  # 测试 文件目录
    parser.add_argument('--outputs-dir', type=str, default="")   #模型 .pth保存目录
    parser.add_argument('--lr', type=float, default=5e-4)   #学习率
    parser.add_argument('--batch-size', type=int, default=1) # 一次处理的图片大小
    parser.add_argument('--num-workers', type=int, default=0)  # 线程数
    parser.add_argument('--num-epochs', type=int, default=400)  #训练次数
    parser.add_argument('--seed', type=int, default=123) # 随机种子
    parser.add_argument("--patience", type=int, default=20, help="how long to wait after last time validation loss improved.")
    args = parser.parse_args()
    #输入与标签图片所在的目录
    args.base_file = r'E:\1_data\training_wv3'
    args.train_file=os.path.join(args.base_file,r'lms')
    args.train_pan=os.path.join(args.base_file,r'pan')
    args.trainlabel_root=os.path.join(args.base_file,r'gt')
    args.eval_file=os.path.join(args.base_file,r'lms_valid')
    args.eval_pan=os.path.join(args.base_file,r'pan_valid')
    args.evallabel_root=os.path.join(args.base_file,r'gt_valid')
    args.outputs_dir=os.path.join(args.base_file,r'outputs')
    args.batch_size=1    #其大小影响BatchNorm2d
    args.num_epochs=400

    # 输出放入固定文件夹里
    args.outputs_dir = os.path.join(args.outputs_dir, r'Pannet_NOPASS')
    # 没有该文件夹就新建一个文件夹
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # benckmark模式，加速计算，但寻找最优配置，计算的前馈结果会有差异
    cudnn.benchmark = True
    cudnn.enabled = True

    # gpu或者cpu模式，取决于当前cpu是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion_edge = nn.MSELoss().to(device)
    #criterion_edge = FFTLoss().to(device)     #shufflemixer weight = 0.1,SAFMN weight = 0.05
    #criterion_edge = FocalFrequencyLoss().to(device)

    

    # 每次程序运行生成的随机数固定
    torch.manual_seed(args.seed)
    start_time=time.time()

    # 构建SRCNN模型，并且放到device上训练

    model = PanNet().to(device)


    # 恢复训练，从之前结束的那个地方开始
    # model.load_state_dict(torch.load('outputs/x3/epoch_173.pth'))


    # 优化函数Adam，lr代表学习率，

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, last_epoch = -1)
    # 预处理训练集
    train_dataset=MyDataset_Pan(args.train_pan,args.train_file, args.trainlabel_root)
    train_dataloader = DataLoader(
        # 数据
        dataset=train_dataset,
        # 分块
        batch_size=args.batch_size,
        # 数据集数据洗牌,打乱后取batch
        shuffle=True,
        # 工作进程，像是虚拟存储器中的页表机制
        num_workers=args.num_workers,
        # 锁页内存，不换出内存，生成的Tensor数据是属于内存中的锁页内存区
        pin_memory=True,
        # 不取余，丢弃不足batchSize大小的图像
        drop_last=True)
    # 预处理验证集
    eval_dataset = MyDataset_Pan(args.eval_pan,args.eval_file, args.evallabel_root)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size)

    # 拷贝权重
    #best_weights = copy.deepcopy(model.state_dict())
    best_model = copy.deepcopy(model)
    best_epoch = 0
    best_ssim = 0.
    best_psnr = 0.
    best_loss = 99999
    counter = 0
    logger = get_logger(os.path.join(args.outputs_dir, 'logger_description.log'))
    logger.info('start training!')

    # 画图用
    lossLog = []
    psnrLog = []

    # 恢复训练
    # for epoch in range(args.num_epochs):
    for epoch in range(1, args.num_epochs + 1):
        # for epoch in range(174, 400):
        # 模型训练入口
        model.train()

        # 变量更新，计算epoch平均损失
        epoch_losses = AverageMeter()

        # 进度条，就是不要不足batchsize的部分
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            # t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs))

            # 每个batch计算一次
            for data in train_dataloader:
                # 对应datastes.py中的__getItem__，分别为lr,hr图像
                pan,inputs, labels = data
                # 梯度清零
                optimizer.zero_grad()
                pan = pan.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 送入模型训练
                outputs = model(pan,inputs)
                #print(labels.size())
                #print(outputs.size())

                # 获得损失
                loss_m = criterion_edge(outputs,labels)
                #loss_grad = grad_criterion(outputs, labels)


                # 显示损失值与长度
                epoch_losses.update(loss_m.item(), len(inputs))


                # 反向传播
                loss_m.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)  #梯度截断
                optimizer.step()
                scheduler.step()

                

                

                # 进度条更新
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                #清除没用的显存
                del inputs,outputs,loss_m
                torch.cuda.empty_cache()
        # 记录lossLog 方面画图
        lossLog.append(np.array(epoch_losses.avg))
        # 可以在前面加上路径
        np.savetxt("mapping_lossLog.txt", lossLog)
        print('train loss: {:.6f}'.format(epoch_losses.avg))

        # save trained parameters
        #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'mapping_epoch_{}.pth'.format(epoch)))
        # save whole models
        torch.save(model,os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # 是否更新当前最好参数
        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        epoch_losses_eval = AverageMeter()

        for data in eval_dataloader:
            pan,inputs, labels = data

            pan = pan.to(device)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 验证不用求导
            with torch.no_grad():
                outputs = model(pan,inputs).clamp(0.0, 1.0)
                 # 获得损失
            loss_m = criterion_edge(outputs,labels)



                # 显示损失值与长度
            epoch_losses_eval.update(loss_m.item(), len(inputs))

            epoch_ssim.update(calc_ssim(outputs, labels), len(inputs))
            
            epoch_psnr.update(calc_psnr(outputs, labels), len(inputs))
            #清除没用的显存
            del inputs,outputs,loss_m
            torch.cuda.empty_cache()
        

        print('eval loss: {:.6f}'.format(epoch_losses_eval.avg))

        print('eval ssim: {:.6f}'.format(epoch_ssim.avg))

        print('eval psnr: {:.6f}'.format(epoch_psnr.avg))

        # 记录psnr
        psnrLog.append(Tensor.cpu(epoch_ssim.avg))
        np.savetxt('mapping_psnrLog.txt', psnrLog)
        
        # 找到更好的权重参数，更新
        if epoch_ssim.avg > best_ssim:
            best_epoch = epoch
            best_ssim = epoch_ssim.avg
            #best_weights = copy.deepcopy(model.state_dict())
            best_model = copy.deepcopy(model)
            optimizer.step()
            counter = 0
        else:
            counter = counter + 1
            print("{0} out of {1}".format(counter,args.patience))
            if counter > args.patience:
                print("early stopping!")
                break
      

        logger.info('Epoch:[{}/{}]\t train loss = {:.6f}\t eval loss = {:.6f}\t eval ssim = {:.6f}\t eval psnr = {:.6f}\t'.format(epoch,args.num_epochs,epoch_losses.avg,epoch_losses_eval.avg,epoch_ssim.avg,epoch_psnr.avg))
        
        print('best epoch: {}, psnr: {:.6f}'.format(best_epoch, best_psnr))

        #torch.save(best_weights, os.path.join(args.outputs_dir, 'mapping_best.pth'))
        torch.save(best_model,os.path.join(args.outputs_dir, 'best.pth'))     
    logger.info('finish!')




    print('best epoch: {}, ssim: {:.6f}'.format(best_epoch, best_psnr))
    end_time=time.time()
    print('Running time: %s Seconds'%(end_time-start_time))