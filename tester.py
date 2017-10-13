import os
from glob import glob
from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from data_loader import get_loader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Tester(object):

    def __init__(self, config, a_data_loader):
        self.config = config

        self.a_data_loader = a_data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.loss = config.loss
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.cnn_type = config.cnn_type

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.build_model()

        if self.num_gpu == 1:
            self.G_AB.cuda()

        elif self.num_gpu > 1:
            self.G_AB = nn.DataParallel(
                self.G_AB.cuda(), device_ids=list(range(self.num_gpu)))

        if self.load_path:
            self.load_model()

    def build_model(self):
        a_height, a_width, a_channel = self.a_data_loader.shape

        if self.cnn_type == 0:
            #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
            conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
        elif self.cnn_type == 1:
            #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
            conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]
        else:
            raise Exception(
                "[!] cnn_type {} is not defined".format(self.cnn_type))

        self.G_AB = GeneratorCNN(a_channel, b_channel, conv_dims,
                                    deconv_dims, self.num_gpu)
        self.G_AB.apply(weights_init)

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [
            int(os.path.basename(path.split('.')[0].split('_')[-1]))
            for path in paths
        ]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        G_AB_filename = '{}/G_AB_{}.pth'.format(self.load_path, self.start_step)
        self.G_AB.load_state_dict(
            torch.load(G_AB_filename, map_location=map_location))
        self.G_BA.load_state_dict(
            torch.load(
                '{}/G_BA_{}.pth'.format(self.load_path, self.start_step),
                map_location=map_location))

        self.D_A.load_state_dict(
            torch.load(
                '{}/D_A_{}.pth'.format(self.load_path, self.start_step),
                map_location=map_location))
        self.D_B.load_state_dict(
            torch.load(
                '{}/D_B_{}.pth'.format(self.load_path, self.start_step),
                map_location=map_location))

        print("[*] Model loaded: {}".format(G_AB_filename))

    def generate_with_A(self, inputs, path, idx=None):
        x_AB = self.G_AB(inputs)
        x_ABA = self.G_BA(x_AB)

        x_AB_path = '{}/{}_x_AB.png'.format(path, idx)
        x_ABA_path = '{}/{}_x_ABA.png'.format(path, idx)

        vutils.save_image(x_AB.data, x_AB_path)
        print("[*] Samples saved: {}".format(x_AB_path))

        vutils.save_image(x_ABA.data, x_ABA_path)
        print("[*] Samples saved: {}".format(x_ABA_path))

    def test(self):
        batch_size = self.config.sample_per_image
        A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)

        test_dir = os.path.join(self.model_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        step = 0
        while True:
            try:
                x_A, x_B = self._get_variable(
                    next(A_loader)), self._get_variable(next(B_loader))
            except StopIteration:
                print("[!] Test sample generation finished. Samples are in {}".
                      format(test_dir))
                break

            vutils.save_image(x_A.data, '{}/{}_x_A.png'.format(test_dir, step))
            vutils.save_image(x_B.data, '{}/{}_x_B.png'.format(test_dir, step))

            self.generate_with_A(x_A, test_dir, idx=step)
            self.generate_with_B(x_B, test_dir, idx=step)

            self.generate_infinitely(
                x_A, test_dir, input_type="A", count=10, nrow=4, idx=step)
            self.generate_infinitely(
                x_B, test_dir, input_type="B", count=10, nrow=4, idx=step)

            step += 1
    ## made by twiiks

    def generate_with_A_retJPG(self, inputs, path, idx=None):
        x_AB = self.G_AB(inputs)

        return x_AB.data

    def testAB(self):
        batch_size = self.config.sample_per_image
        A_loader = iter(self.a_data_loader)

        x_A = self._get_variable(next(A_loader))

        x_AB = self.generate_with_A_retJPG(x_A, test_dir, idx=step)
        return x_AB
    
    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
