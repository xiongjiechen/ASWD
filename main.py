from __future__ import print_function
from TransformNet import TransformNet,Mapping,Mapping_Conv

from DCGANAE import DCGANAE,Discriminator

import imageio
import torch
import argparse
import os
from torch import  optim
import torch.nn as nn
from torchvision import transforms
import numpy as np
from experiments import sampling
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
import random
import time
from fid_score import *
from utils import circular_function,train_net
# torch.backends.cudnn.enabled = False
class CustomDataset(Dataset):
    def __init__(self, root, image_loader=io.imread, transform=None):
        self.root = root


        self.images_files = os.listdir(root)
        self.loader = image_loader
        self.transform = transform

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return len(self.images_files)

    def __getitem__(self, index):
        img = Image.fromarray(self.loader(self.root+"/"+self.images_files[index]))

        if self.transform is not None:
            images = self.transform(img)
        return (images,1)
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train',image_loader=io.imread):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.loader = image_loader
        self.files_A = sorted(os.listdir(os.path.join(root, '%sA' % mode)))
        self.files_B = sorted(os.listdir(os.path.join(root, '%sB' % mode)))
        self.rootA = os.path.join(root, '%sA' % mode)
        self.rootB = os.path.join(root, '%sB' % mode)
    def __getitem__(self, index):
        item_A=self.transform(Image.open(self.rootA + "/" + self.files_A[index % len(self.files_A)]).convert('RGB'))
        if self.unaligned:
            item_B=self.transform(Image.open(self.rootB + "/" + self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.rootB + "/"+self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
def main():
    # train args
    parser = argparse.ArgumentParser(description='Augmented Sliced Wasserstein Autoencoder')
    parser.add_argument('--datadir', default='./', help='path to dataset')
    parser.add_argument('--outdir', default='./result/',
                        help='directory to output images')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--num-workers', type=int, default=16, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 16)')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 16)')
    parser.add_argument('--g', type=str, default='circular',
                        help='g')
    parser.add_argument('--num-projection', type=int, default=1000,
                        help='number projection')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='Regularization strength')
    parser.add_argument('--p', type=int, default=2,
                        help='Norm p')
    parser.add_argument('--niter', type=int, default=5,
                        help='number of iterations')
    parser.add_argument('--r', type=float, default=1000,
                        help='R')
    parser.add_argument('--latent-size', type=int, default=32,
                        help='Latent size')
    parser.add_argument('--dataset', type=str, default='CIFAR',
                        help='(CELEBA|CIFAR)')
    parser.add_argument('--model-type', type=str, required=True,
                        help='(ASWD|SWD|MSWD|DSWD|GSWD|)')
    parser.add_argument('--gpu',type=str,required=False,default=0)
    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    if (args.g == 'circular'):
        g_function = circular_function
    model_type = args.model_type
    latent_size = args.latent_size
    num_projection = args.num_projection
    dataset=args.dataset
    model_dir = os.path.join(args.outdir, model_type)
    assert dataset in ['CELEBA', 'CIFAR']
    assert model_type in ['ASWD', 'SWD', 'MSWD', 'DSWD', 'GSWD']
    if not (os.path.isdir(args.datadir)):
        os.makedirs(args.datadir)
    if not (os.path.isdir(args.outdir)):
        os.makedirs(args.outdir)
    if not (os.path.isdir(model_dir)):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")
    print('batch size {}\nepochs {}\nAdam lr {} \n using device {}\n'.format(
        args.batch_size, args.epochs, args.lr, device.type
    ))

    if (dataset == 'CIFAR'):
        from DCGANAE import Discriminator
        image_size = 64
        num_chanel = 3
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.datadir, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    elif (dataset =='CELEBA'):
        from DCGANAE import Discriminator
        image_size = 64
        num_chanel = 3
        dataset = CustomDataset(root=args.datadir+'img_align_celeba',
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        # Create the dataloader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers,pin_memory=True)



    model = DCGANAE(image_size=64, latent_size=latent_size, num_chanel=3, hidden_chanels=64, device=device).to(
        device)
    #model=nn.DataParallel(model)
    #model.to(device)
    dis = Discriminator(64, args.latent_size, 3, 64).to(device)
    disoptimizer = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if (model_type == 'DSWD' or model_type == 'DGSWD'):
        transform_net = TransformNet(64 * 8 * 4 * 4).to(device)
        op_trannet = optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
        train_net(64 * 8 * 4 * 4, 1000, transform_net, op_trannet,device=device)

    if model_type=='ASWD':
        phi=Mapping(64 * 8 * 4 * 4).to(device)
        phi_op = optim.Adam(phi.parameters(), lr=0.001, betas=(0.5, 0.999))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    epoch_cont=0
    generated_sample_number=64
    fixednoise= torch.randn((generated_sample_number, latent_size)).to(device)
    loss_recorder=[]
    generated_sample_size=64
    W2_recorder=np.zeros([41,40])
    save_idx=str(time.time()).split('.')
    save_idx=save_idx[0]+save_idx[1]
    path_0=model_dir+'/'+args.dataset+'/fid/'+save_idx
    os.mkdir(path_0)
    if args.dataset=='CIFAR':
        interval_=10
        fid_stats_file=args.datadir+'/fid_stats_cifar10_train.npz'
    else:
        interval_=5
        fid_stats_file=args.datadir+'/fid_stats_celeba.npz'
    fid_recorder=np.zeros(args.epochs//interval_+1)
    for epoch in range(epoch_cont,args.epochs):
        total_loss=0.0
        
        for batch_idx, (data, y) in tqdm(enumerate(train_loader, start=0)):
            if(model_type=='SWD'):
                loss= model.compute_loss_SWD(dis,disoptimizer,data,torch.randn,num_projection,p=args.p,epoch=epoch,batch_idx=batch_idx)
            elif (model_type=='GSWD'):
                loss = model.compute_loss_GSWD(dis, disoptimizer, data, torch.randn,g_function,args.r, num_projection, p=args.p)
            elif(model_type=='MSWD'):
                loss,v = model.compute_loss_MSWD(dis, disoptimizer,data,torch.randn,p=args.p,max_iter=args.niter)
            elif (model_type == 'DSWD'):
                loss = model.compute_lossDSWD(dis,disoptimizer,data,torch.randn,num_projection,transform_net,op_trannet,p=args.p,max_iter=args.niter,lam=10)
            elif (model_type == 'DGSWD'):
                loss = model.compute_lossDGSWD(dis,disoptimizer,data,torch.randn,num_projection,transform_net,op_trannet,g_function,args.r,p=args.p,max_iter=args.niter,lam=10)
            elif (model_type == 'SINKHORN'):
                loss = model.compute_loss_sinkhorn(dis, disoptimizer, data, torch.randn, p=2, n_iter=100,e=100)
            elif (model_type == 'CRAMER'):
                loss = model.compute_loss_cramer(dis,disoptimizer,data, torch.randn)
            elif model_type=='ASWD':
                loss= model.compute_lossASWD(dis, disoptimizer, data, torch.randn, num_projection, phi, phi_op, p=2,
                                               max_iter=args.niter, lam=args.lam,epoch=epoch,batch_idx=batch_idx)
            optimizer.zero_grad()
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()
        if epoch==0 or (epoch+1)%interval_==0:
            path_1=path_0+'/'+str('%03d'%epoch)
            os.mkdir(path_1)
            for j in range(20):
                fixednoise_=torch.randn((1000,32)).to(device)
                imgs=model.decoder(fixednoise_)

                for i,img in enumerate(imgs):
                    img=img.transpose(0,-1).transpose(0,1).cpu().detach().numpy()
                    img=(img*255).astype(np.uint8)
                    imageio.imwrite(path_1+'/'+args.model_type+'_'+str(args.num_projection)+'_'+str('%03d'%epoch)+'_'+str(1000*j+i)+'.png', img)
            fid_value = calculate_fid_given_paths([path_1+'/',fid_stats_file],50,True,2048)
            fid_recorder[(epoch+1)//interval_]=fid_value
            np.save(path_0+'/fid_recorder_'+'np_'+str(num_projection)+'.npy',fid_recorder)
            print('fid score:', fid_value)
            os.system("rm -rf "+path_1)
        total_loss/=(batch_idx+1)
        loss_recorder.append(total_loss)
        print("Epoch: "+str(epoch)+" Loss: "+str(total_loss))


        if(epoch%1==0 or epoch==args.epochs-1):
            sampling(model_dir+'/'+args.dataset+'/sample_epoch_'+str(epoch)+".png",fixednoise,model.decoder,generated_sample_number,generated_sample_size,num_chanel)
            torch.save(model.state_dict(),args.outdir+args.dataset+'/'+model_type+'_'+str(args.batch_size)+'_'+str(num_projection)+'_'+str(latent_size)+'_model.pth')
            torch.save(dis.state_dict(),args.outdir+args.dataset+'/'+model_type+'_'+str(args.batch_size)+'_'+str(num_projection)+'_'+str(latent_size)+'_discriminator.pth')
    np.save(args.outdir+args.dataset+'/'+model_type+'_'+str(args.batch_size)+'_'+str(num_projection)+'_'+str(latent_size)+'_loss.npy',loss_recorder)
if __name__ == '__main__':
    main()
