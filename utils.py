import os
import numpy as np
import scipy.stats
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import json
import random
from einops import rearrange

def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)

# the function for printing neural network's weights
print_weight_cnt = 0
def print_weights(data : torch.Tensor, mode='a'):
    global print_weight_cnt
    if print_weight_cnt == 0:
        fout = open('weight.txt', 'w')  # just open-close to clear previous content
        fout.close()
    print_weight_cnt += 1
    buffer_str = str(data.cpu().detach().numpy().copy())
    with open('weight.txt', mode) as fout:
        fout.write('=============times {}=============\n'.format(print_weight_cnt))
        fout.write(str(buffer_str))
        fout.write('\n')
        fout.close()

def image_normalize(image, denormalize=False):
    '''
    image: H x W x C, image pixel's value range should be 0~1
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if denormalize == False:  # noralize
        for channel in range(3):
            image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    else:  # de-normalize
        for channel in range(3):
            image[:, :, channel] = image[:, :, channel] * std[channel] + mean[channel]
    return image

def make_grid_images(tensor_image, denormalize=True, save_path=None):
    '''
    :param tensor: B x C x H x W, the pixel value should be ranges from 0~1
    :param denormalize:
    :param save_path:
    :return: grid_image, H' x W' x 3
    '''
    # vmax, vmin = torch.max(tensor_image), torch.min(tensor_image)
    image_temp = torch.clone(tensor_image)
    # de-normalize
    if denormalize == True:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for channel in range(3):
            image_temp[:, channel, :, :] = image_temp[:, channel, :, :] * std[channel] + mean[channel]
    grid_image = torchvision.utils.make_grid(image_temp, scale_each=0.2)  # 3 x H x W, make_grid will output image with 3 channels
    grid_image = grid_image.permute(1, 2, 0)  # H x W x 3
    if save_path != None:
        grid_image = grid_image.cpu().detach().numpy()[:, :, ::-1]  # convert RGB image to BGR image
        cv2.imwrite(save_path, grid_image * 255)  # convert 0~1 to be 0~255

    return grid_image


def make_uncertainty_map(sigmas_np, B):
    # sigmas: (B*N) * (L*L), numpy
    # we want to make B images and each image shows N keypoint hotspots.
    W, H = 368, 368
    sigmas = sigmas_np  # sigmas_tensor.cpu().detach().numpy()
    N = int(sigmas.shape[0] / B)
    L = int(np.sqrt(sigmas.shape[1]))
    im_combined = np.zeros((B, H, W))
    for im_j in range(B):
        for i in range(im_j * N, (im_j + 1) * N):
            im = sigmas[i, :].reshape(L, L)
            im_resize = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
            vmin, vmax = np.min(im_resize), np.max(im_resize)
            # print(vmin, vmax)
            if vmax > vmin:
                # im_resize = (im_resize - vmin) / (vmax-vmin)  # normalize to 0~1
                im_resize = (im_resize - vmin) / (vmax)
            # merging using max operation
            ind = im_combined[im_j] < im_resize
            im_combined[im_j][ind] = im_resize[ind]
    vmin, vmax = np.min(im_combined), np.max(im_combined)
    # plt.imshow(im_combined[0])
    # plt.show()

    return im_combined  # B x H x W


def save_plot_image(im, save_path, does_show=False):
    '''
    :param im: H x W x 3 or H x W , numpy
    :return:
    '''
    H, W = im.shape[0], im.shape[1]
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = fig.gca()
    fig.tight_layout()
    fig.patch.set_alpha(0.)  # set the figure face to be transparent
    # im = Image.open("/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images/dog/do86.jpeg").convert('RGB')
    # im = im.resize((square_image_length, square_image_length), PIL.Image.BILINEAR)
    # plt.imshow(im, cmap=plt.cm.jet)
    plt.imshow(im, cmap=plt.cm.viridis)  # default
    # plt.imshow(im)
    # plt.show()

    # remove ticks but the frame still exists
    plt.xticks([])
    plt.yticks([])
    ax.invert_yaxis()

    # Remove the white margin around image
    fig.set_size_inches(W / 100.0, H / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(save_path)
    if does_show == True:
        plt.show()

def compute_eigenvalues(covar):
    '''
    :param covar: 2 x 2
    :return: eigenvalues, eigenvectors, orientation
    '''
    # eigenvalues e: 2 x 2, each row is an eigenvalue e[i,0]+j*e[i, 1],
    # eigenvectors v: 2 x 2, each column is a corresponding eigenvector v[:, i]
    e, v = torch.eig(covar, eigenvectors=True)
    _, indices = torch.sort(e[:, 0], descending=True, dim=0)
    e2, v2 = e[indices, :], v[:, indices]
    radian = torch.atan2(v2[1, 0], v2[0, 0])  # atan2(vy, vx)
    angle = radian / 3.1415926 * 180  # orientation of the eigenvector for major axis
    return e2, v2, angle

def mean_confidence_interval(accs, confidence=0.95):
    '''
    compute mean and standard error of mean for a sequence of observations
    using t-test
    '''
    if isinstance(accs, np.ndarray) == False:
        accs = np.array(accs)

    n = accs.shape[0]
    if n == 1:
        return accs[0], 0
    m, se = np.mean(accs), scipy.stats.sem(accs)  # sem = standard error of mean = sigma / sqrt(n)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)  # ppf here is the inverse of cdf (cumulative distributin function)
    return m, h

def mean_confidence_interval_multiple(accs_multiple, confidence=0.95):
    '''
    accs_multiple: K x N, K rows, each row will compute mean_confidence_interval
    '''
    K = len(accs_multiple)
    mean, interval = np.zeros(K), np.zeros(K)
    for i in range(K):
        mean[i], interval[i] = mean_confidence_interval(np.array(accs_multiple[i]), confidence=confidence)

    return mean, interval

def load_samples(ann_json_files, local_json_root):
    '''
    ann_json_files: a list
    local_json_root: a path
    return: a list of samples
    '''
    samples = []
    for p in ann_json_files:
        annotation_path = os.path.join(local_json_root, p)
        with open(annotation_path, 'r') as fin:
            # self.samples = json.load(fin)
            samples_temp = json.load(fin)
            # self.samples = dataset['anns']
            fin.close()
        samples += samples_temp

    return samples

def power_norm1(x, SIGMA):
    out = 2/(1 + torch.exp(-SIGMA*x)) - 1
    return out

def power_norm2(x, SIGMA):
    out = torch.sign(x) * torch.abs(x).pow(SIGMA)
    return out





def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



# count model learnable parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000



def get_patches(ims: torch.Tensor, patch_size=(32, 32), save=False, prefix='s', saveroot='./episode_images/patched_ims'):
    '''
    ims: cpu Tensor, B x C x H x W
    return patched_ims: B x (grid_h*grid_w) x C x p1 x p2
    '''
    if save:
        if os.path.exists(saveroot) == False:
            os.makedirs(saveroot)
    B, _, H, W = ims.shape
    patched_ims = rearrange(ims, 'B C (h p1) (w p2) -> B C (h w) p1 p2', p1=patch_size[0], p2=patch_size[1])
    patched_ims = patched_ims.permute(0, 2, 1, 3, 4)  # B x (grid_h*grid_w) x C x p1 x p2
    if save:
        for i in range(B):
            # grid_iamge: C x H' x W'
            grid_image = torchvision.utils.make_grid(patched_ims[i], nrow=W // patch_size[1], padding=2, normalize=False, pad_value=0.8)
            grid_image = grid_image.permute(1, 2, 0)
            grid_image = grid_image.numpy()[:, :, ::-1]
            cv2.imwrite(os.path.join(saveroot, prefix+'_'+str(i)+'.jpg'), grid_image * 255)

    return patched_ims


def ele_max(a, b=0):
    # sign = (a >= b).detach().float()
    # return sign * a + (1 - sign) * b
    return torch.clamp(a, min=b)

def display_all_args(args):
    '''
    args: the parsed args, type(args) == argparse.Namespace
    '''
    print('Display all the hyper-parameters in args:')
    for arg in vars(args):
        value = getattr(args, arg)
        # if value is not None:
        print('%s: %s' % (str(arg), str(value)))
    print('------------------------')

#==============================================================
# Below code is useless
def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam', 'sgd'])
    parser.add_argument("--lr", help="initial learning rate", type=float)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int)
    parser.add_argument("--stage", help="number lr stages", type=int)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float)
    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=20)
    parser.add_argument("--resnet", help="whether use resnet18 as backbone or not", action="store_true")

    ## PN model related hyper-parameters
    parser.add_argument("--alpha", help="scalar for pose loss", type=int)
    parser.add_argument("--num_part", help="number of parts", type=int)
    parser.add_argument("--percent", help="percent of base images with part annotation", type=float)

    ## shared optional
    parser.add_argument("--batch_size", help="batch size", type=int)
    parser.add_argument("--load_path", help="load path for dynamic/transfer models", type=str)

    args = parser.parse_args()

    if args.resnet:
        name = 'ResNet18'
    else:
        name = 'Conv4'

    return args, name



