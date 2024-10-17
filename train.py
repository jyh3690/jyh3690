import os
import json
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from tqdm import tqdm
from dataset import SurfaceDefectDataset
from dataset import defect_labels
from resnet18 import SurfaceDectectResNet
import argparse
from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from math import cos, pi
import numpy as np 

class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=10):
        """Initializes simple early stopping mechanism for YOLOv5, with adjustable patience for non-improving epochs."""
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """Evaluates if training should stop based on fitness improvement and patience, returning a boolean."""
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        # if stop:
        #     LOGGER.info(
        #         f"Stopping training early as no improvement observed in last {self.patience} epochs. "
        #         f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
        #         f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
        #         f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
        #     )
        return stop


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda:0', help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--batch_size", type=int, default=4, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--train_path",type=str,default='E:\\nano\\1_projects\\code\develop\\classfy\\data\\mobilenet\\data3\\train',
                        help="train img path")
    parser.add_argument("--val_path",type=str,default='E:\\nano\\1_projects\\code\develop\\classfy\\data\\mobilenet\\data1\\val',
                        help="val img path")
    parser.add_argument("--num_classes",type=int,default=5,help="classes number")
    parser.add_argument('--model', default='mobilenet_v3_small', type=str, metavar='MODEL',
                        help='Name of model to train,mobilenet_v3_small mobilenet_v3_large or resnet18')
    parser.add_argument("--save_path",type=str,
                        default='E:\\nano\\1_projects\\code\\develop\\classfy\\pth\\mobilenetv3\\mobilenet_bs4_lr0.1_last.pth')
    parser.add_argument("--weights", type=str, 
                        default="E:\\nano\\1_projects\\code\\develop\\classfy\\pth\\mobilenetv3\\mobilenet_bs4_lr0.1_last_0.874.pth", help="initial weights path")
    parser.add_argument("--optimizer", type=str, 
                        choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-decay', type=str, default='step',
                        help='mode for learning rate decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay') #L2正则化，防止过拟合
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--warmup', action='store_true',
                        help='set lower initial learning rate to warm up the training') #在命令行指定该参数则为true，否则false
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--patience", type=int, default=50, help="EarlyStopping patience (epochs without improvement)")
    return parser.parse_args()

def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    """
    Initializes YOLOv5 smart optimizer with 3 parameter groups for different decay configurations.

    Groups are 0) weights with decay, 1) weights no decay, 2) biases no decay.
    """
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    # LOGGER.info(
    #     f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
    #     f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    # )
    return optimizer


def adjust_learning_rate(opt,optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if opt.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = opt.epochs * num_iter

    if opt.lr_decay == 'step':
        lr = opt.lr * (opt.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif opt.lr_decay == 'cos':
        lr = opt.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif opt.lr_decay == 'linear':
        lr = opt.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif opt.lr_decay == 'schedule':
        count = sum([1 for s in opt.schedule if s <= epoch])
        lr = opt.lr * pow(opt.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_decay))

    if epoch < warmup_epoch:
        lr = opt.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(opt,train_loader,model,optimizer,loss_fn,iter_epoch):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        adjust_learning_rate(opt,optimizer,iter_epoch,step,len(train_loader))
        images, labels = data['image'], data['label']
        optimizer.zero_grad()
        if images.size(0) == 1:  # 处理批次大小为1的情况
            continue
        else:
            # model.train()  # 正常训练模式
            outputs = model(images.to(opt.device))
        loss = loss_fn(outputs, labels.to(opt.device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # train_loss.append(loss.item())
        train_bar.desc = f'train epoch[{iter_epoch + 1}/{opt.epochs}] loss:{loss:.3f}'

def validate(opt,validate_loader,model):
    model.eval() 
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data['image'], val_data['label']
            outputs = model(val_images.to(opt.device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(opt.device)).sum().item()
    return acc

RANK = int(os.getenv("RANK", -1))
def main(opt):

    print(f'using {opt.device}')
    
    seed = opt.seed + RANK+1
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    #数据加载
    train_dataset = SurfaceDefectDataset(opt.train_path) 
    cla_dict = dict((i, label) for i, label in enumerate(defect_labels)) # 类别和index的对应关系写入文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
 
    nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8])
    print(f'using {nw} dataloader workers every process')
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=nw)
    validate_dataset = SurfaceDefectDataset(opt.val_path)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=opt.batch_size,shuffle=True,num_workers=nw)
                               
    # 训练
    if opt.model == "mobilenet_v3_small":
       net = MobileNetV3_Small(opt.num_classes)
    elif opt.model == "mobilenet_v3_large":
        net = MobileNetV3_Large(opt.num_classes)
    elif  opt.model == "resnet18":
        net = SurfaceDectectResNet(opt.num_classes)

    pretrained = opt.weights.endswith(".pth")
    if pretrained:
        net.load_state_dict(torch.load(opt.weights,map_location=opt.device,weights_only=True))
    #冻结某些层，主要适用于fine-tuning或迁移学习
    freeze = opt.freeze
    if freeze: 
        '''
        resnet18的微调冻结固定层未调试
        freeze = [f"cnn_layers.layer.{x}.{x}.conv{x}.weight" for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  
        for k, v in net.named_parameters():
            print(f"Freeze list: {freeze}")
            print(f"Parameter name: {k}")
            v.requires_grad = True  # 训练所有层
            if any(freeze_layer in k for freeze_layer in freeze):
                print(f"freezing {k}")
                v.requires_grad = False  # 冻结匹配的层
        '''
        freeze = [f"bneck.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # 假设冻结 bneck 层
        #  ["conv1.weight"]+["bn1.weight"]+["bn1.bias"]
        for k, v in net.named_parameters():
            # print(f"parameter name: {k}")
            v.requires_grad = True  # 训练所有层
            if any(freeze_layer in k for freeze_layer in freeze):
                # print(f"freezing {k}")
                v.requires_grad = False  # 冻结匹配的层
    net.to(opt.device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = smart_optimizer(net, opt.optimizer, opt.lr, opt.momentum, opt.weight_decay)
    optimizer = torch.optim.SGD(net.parameters(),opt.lr,opt.weight_decay,opt.momentum,nesterov=False)
    best_acc = 0.0
    train_steps = len(train_loader)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # train_loss=[]
    for epoch in range(opt.epochs): 
        train(opt,train_loader,net,optimizer,loss_fn,epoch)
        
        acc = validate(opt,validate_loader,net)
        val_accuracy = acc / val_num
        print(f'[epoch {epoch + 1} ,'f'val_accuracy:{val_accuracy:.3f}]')
        stop = stopper(epoch=epoch, fitness=val_accuracy)
        # print(f'[epoch {epoch + 1} train_loss: {running_loss / train_steps:.3f},'
        #         f'val_accuracy:{val_accuracy:.3f}]')

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), opt.save_path)
        
        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            print("early stop!")
            break  # must break all DDP ranks
    print(f'best_acc:{best_acc:.3f}')
if __name__ == '__main__':
    starttime = time.time()
    opt=parse_opt()
    main(opt)
    endtime = time.time()
    print('train total time: {:.5f} s'.format(endtime-starttime))
