# Augmented-Sliced-Wasserstein-Distances

This repository provides the code to reproduce the experimental results in the paper **[Augmented Sliced Wasserstein Distances](https://arxiv.org/abs/2006.08812)** by **[Xiongjie Chen](https://github.com/xiongjiechen), [Yongxin Yang](https://yang.ac/)** and **[Yunpeng Li](https://www.surrey.ac.uk/people/yunpeng-li)**.
## Prerequisites

### Python packages

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```

### Datasets
Two datasets are used in this repository, namely the [CIFAR10](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf) dataset and [CELEBA](http://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html) dataset.
- The CIFAR10 dataset (64x64 pixels) will be automatically downloaded from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz when running the experiment on CIFAR10 dataset. 
- The CELEBA dataset needs be be manually downloaded and can be found on the website http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, we use the cropped CELEBA dataset with 64x64 pixels.

### Precalculated Statistics

To calculate the [Fréchet Inception Distance (FID score)](https://arxiv.org/abs/1706.08500), precalculated statistics for datasets

- [CIFAR 10](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) (calculated on all training samples)
- [cropped CelebA](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz) (64x64, calculated on all samples)

are provided at: http://bioinf.jku.at/research/ttur/.
## Project & Script Descriptions
Two experiments are included in this repository, where benchmarks are from the paper [Generalized Sliced Wasserstein Distances](http://papers.nips.cc/paper/8319-generalized-sliced-wasserstein-distances) and the paper [Distributional Sliced-Wasserstein and Applications to Generative Modeling](https://arxiv.org/pdf/2002.07367.pdf), respectively. The first one is on the task of sliced Wasserstein flow, and the second one is on generative modellings with GANs. For more details and setups, please refer to the original paper **[Augmented Sliced Wasserstein Distances](https://arxiv.org/abs/2006.08812)**.
### Directories
- ```./result/ASWD/CIFAR/``` contains generated imgaes trained with the ASWD on CIFAR10 dataset.
- ```./result/ASWD/CIFAR/fid/``` FID scores of generated imgaes trained with the ASWD on CIFAR10 dataset are saved in this folder.
- ```./result/CIFAR/``` model's weights and losses in the CIFAR10 experiment are stored in this directory.

Other setups follow the same naming rule.
### Scripts
The sliced Wasserstein flow example can be found in the [jupyter notebook](https://github.com/xiongjiechen/ASWD/blob/master/Sliced%20Waaserstein%20Flow.ipynb).

The following scripts belong to the generative modelling example:
- [main.py](https://github.com/xiongjiechen/ASWD/blob/master/main.py) : run this file to conduct experiments.
- [utils.py](https://github.com/xiongjiechen/ASWD/blob/master/utils.py) : contains implementations of different sliced-based Wasserstein distances.
- [TransformNet.py](https://github.com/xiongjiechen/ASWD/blob/master/TransformNet.py) : edit this file to modify architectures of neural networks used to map samples. 
- [experiments.py](https://github.com/xiongjiechen/ASWD/blob/master/experiments.py) : functions for generating and saving randomly generated images.
- [DCGANAE.py](https://github.com/xiongjiechen/ASWD/blob/master/DCGANAE.py) : neural network architectures and optimization objective for training GANs.
- [fid_score.py](https://github.com/xiongjiechen/ASWD/blob/master/fid_score.py) : functions for calculating statistics (mean & covariance matrix) of distributions of images and the FID score between two distributions of images.
- [inception.py](https://github.com/xiongjiechen/ASWD/blob/master/inception.py) : download the pretrained [InceptionV3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) model and generate feature maps for FID evaluation.

## Experiment options for the generative modelling example
The generative modelling experiment evaluates the performances of GANs trained with different sliced-based Wasserstein metrics. To train and evaluate the model, run the following command:

```
python main.py  --model-type ASWD --dataset CIFAR --epochs 200 --num-projection 1000 --batch-size 512 --lr 0.0005
```

### Basic parameters
- ```--model-type``` type of sliced-based Wasserstein metric used in the experiment, available options: ASWD, DSWD, SWD, MSWD, GSWD. Must be specified.
- ```--dataset``` select from: CIFAR, CELEBA, default as CIFAR.
- ```--epochs``` training epochs, default as 200.
- ```--num-projection``` number of projections used in distance approximation, default as 1000.
- ```--batch-size``` batch size for one iteration, default as 512.
- ```--lr``` learning rate, default as 0.0005.

### Optional parameters

- ```--niter``` number of iteration, available for the ASWD, MSWD and DSWD, default as 5.
- ```--lam``` coefficient of regularization term, available for the ASWD and DSWD, default as 0.5.
- ```--r``` parameter in the circular defining function, available for GSWD, default as 1000.
## Experimental results
<details>
<summary> Sliced Wasserstein flow </summary>
We conduct the sliced Wasserstein flow experiment on eight different datasets and the experimental results are presented in the following figure. The first and third columns in the figure below are target distributions. The second and fourth columns are log 2-Wasserstein distances between the target distribution and the source distribution. The horizontal axis show the number of training iterations. Solid lines and shaded areas represent the average values and 95% confidence intervals of log 2-Wasserstein distances over 50 runs.

![test](https://github.com/xiongjiechen/ASWD/blob/master/images/swf.PNG)

</details>

<details>
<summary> Generative modelling </summary>

The table below provides FID scores of generative models trained with different distance metrics. Lower scores indicate better image qualities. In what follows, *L* is the number of projections, we run each experiment 10 times and report the average values and standard errors of FID scores for CIFAR10 dataset and CELEBA dataset. The running time per training iteration for one batch containing 512 samples is computed based on a computer with an Intel (R) Xeon (R) Gold 5218 CPU 2.3 GHz and 16GB of RAM, and a RTX 6000 graphic card with 22GB memories.

![test](https://github.com/xiongjiechen/ASWD/blob/master/images/GANs_tab.PNG)

With *L*=1000 projections, the following figure shows the convergence rate of FID scores of generative models trained with different metrics on CIFAR10 and CELEBA datasets. The error bar represents the standard deviation of the FID scores at the specified training epoch among 10 simulation runs.

![test](https://github.com/xiongjiechen/ASWD/blob/master/images/GANs_fig.PNG)
</details>

## References 
### Code
The code of generative modelling example is based on the implementation of [DSWD](https://github.com/VinAIResearch/DSW) by [VinAI Research](https://github.com/VinAIResearch).

The pytorch code for calculating the FID score is from https://github.com/mseitzer/pytorch-fid.

### Papers
- [Distributional Sliced-Wasserstein and Applications to Generative Modeling](https://arxiv.org/pdf/2002.07367.pdf)
- [Generalized Sliced Wasserstein Distances](http://papers.nips.cc/paper/8319-generalized-sliced-wasserstein-distances)
- [Sliced Wasserstein Auto-Encoders](https://openreview.net/forum?id=H1xaJn05FQ)
- [Max-Sliced Wasserstein Distance and its Use for GANs](http://openaccess.thecvf.com/content_CVPR_2019/html/Deshpande_Max-Sliced_Wasserstein_Distance_and_Its_Use_for_GANs_CVPR_2019_paper.html)
## Citation
If you find this code useful for your research, please cite our paper:
```
@article{chen2020augmented,
    title={Augmented Sliced Wasserstein Distance},
    author={Chen, Xiongjie and Yang, Yongxin and Li, Yunpeng},
    journal={arXiv preprint arXiv:2006.08812},
    year={2020}
}
```
