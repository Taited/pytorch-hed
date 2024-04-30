#!/usr/bin/env python
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose, Resize
)
import os
import os.path as osp
import tqdm
import cv2

# requires at least pytorch version 1.3.0
assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13)
# do not compute gradients for computational performance
torch.set_grad_enabled(False) 
# use cudnn for computational performance
torch.backends.cudnn.enabled = True


def parse_args():
    parser = argparse.ArgumentParser(description='Inference HED')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use ')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='id of gpu to use ')
    parser.add_argument(
        '--input-path',
        type=str,
        default='dataset/LSUN/church_outdoor/train/src')
    parser.add_argument(
        '--output-path',
        type=str,
        default='dataset/LSUN/church_outdoor/train/edge')
    
    args = parser.parse_args()
    return args
    

class Normalize:
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, x):
        x = np.array(x)
        for i in range(3):
            x[:, :, i] = x[:, :, i] - self.mean[i]
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(2, 0, 1)
        return x


class HighContrast(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self):
		pass

	def __call__(self, img: Image.Image):
		img = np.array(img)
		lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
		# 将LAB格式分割为L、A和B通道
		l, a, b = cv2.split(lab)
		# 创建CLAHE对象并应用于L通道
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl = clahe.apply(l)
		# 合并L、A和B通道
		merged = cv2.merge([cl, a, b])
		# 将LAB格式转换回BGR格式
		result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
		return np.array(result)

    
class DummyDataset(Dataset):
    def __init__(self, data_root,
                 transform=Compose([
                     Resize([320, 480]),
                     HighContrast(),
                     Normalize([104.00698793, 
                                116.66876762, 
                                122.67891434])])):
        super().__init__()
        self.data_root = data_root
        self.data_info = self.__prepare_data_info()
        self.length = len(self.data_info)
        self.transform = transform
        print(f"{self.length} samples were loaded")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        path = self.data_info[index]
        img = Image.open(path)
        ori_shape = img.size
        if self.transform:
            img = self.transform(img)
        results = {'img': img,
                   'ori_size': ori_shape,
                   'name': osp.basename(path)}
        return results
    
    def __prepare_data_info(self):
        data_info = []
        for file_name in os.listdir(self.data_root):
            data_info.append(
                osp.join(self.data_root, file_name))
        return data_info


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict(
            {strKey.replace('module', 'net'): 
                tenWeight for strKey, tenWeight in 
                torch.hub.load_state_dict_from_url(
                    url='http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch', 
                        file_name='hed-bsds500').items() })
    @torch.no_grad()
    def forward(self, tenInput):
        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = F.interpolate(input=tenScoreOne, size=tenInput.shape[2:], mode='bilinear', align_corners=False)
        tenScoreTwo = F.interpolate(input=tenScoreTwo, size=tenInput.shape[2:], mode='bilinear', align_corners=False)
        tenScoreThr = F.interpolate(input=tenScoreThr, size=tenInput.shape[2:], mode='bilinear', align_corners=False)
        tenScoreFou = F.interpolate(input=tenScoreFou, size=tenInput.shape[2:], mode='bilinear', align_corners=False)
        tenScoreFiv = F.interpolate(input=tenScoreFiv, size=tenInput.shape[2:], mode='bilinear', align_corners=False)

        combined = self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))
        
        results = {
            'stage_1': tenScoreOne,
            'stage_2': tenScoreTwo,
            'stage_3': tenScoreThr,
            'stage_4': tenScoreFou,
            'stage_5': tenScoreFiv,
            'combine': combined
        }
        return results


def save_imgs(img: torch.Tensor, 
              ori_size,
              save_root: str,
              save_name: str):
    img = img.clip(0.0, 1.0).squeeze(1) * 255.0
    for i in range(img.shape[0]):
        w = ori_size[0][i]
        h = ori_size[1][i]
        save_img = img[i, :, :].cpu().numpy()
        save_img = Image.fromarray(save_img.astype(np.uint8))
        save_img = save_img.resize((w, h))
        save_img.save(osp.join(save_root, save_name[i]))
    
    
if __name__ == '__main__':
    args = parse_args()   # 输入输出，device等
    if not osp.exists(args.output_path):
        os.mkdir(args.output_path)
    
    torch.cuda.set_device(args.gpu_id)
    model = Network().cuda().eval()   ##model从CPU to GPU,model中随机参数固定,评估
    
    if not osp.isdir(args.input_path):
        assert f"The input path should be a " \
            + "directory instead of {args.input_path}"
    
    # create dataloader
    dataset = DummyDataset(args.input_path)
    dataloader = DataLoader(dataset, args.batch_size,
                            shuffle=False, num_workers=8,
                            pin_memory=True, drop_last=False)
    
    with tqdm.tqdm(total=len(dataloader)) as pbar:
        for batch_id, batch_data in enumerate(dataloader):
            results = model(batch_data['img'].cuda())['combine']
            save_imgs(results, batch_data['ori_size'],
                      args.output_path, 
                      batch_data['name'])
            pbar.update(1)
