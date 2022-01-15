"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, mean=1., std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        #num_bottleneck = input_dim # We remove the input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck, affine=True)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.ave_pool = nn.AdaptiveMaxPool2d((1,1))
        self.classifier = classifier
    def forward(self, x, ave_pool = False):
        if ave_pool:
            x = self.ave_pool(x)
        x = self.add_block(x)# [B, 512]
        x = self.classifier(x)
        return x

class domain_discriminator(nn.Module):
    
	def __init__(self, rp_size, optimizer, lr, momentum, weight_decay,n_outputs=512):
		super(domain_discriminator, self).__init__()

		self.domain_discriminator = nn.Sequential()
		self.domain_discriminator.add_module('d_fc1', nn.Linear(rp_size, 512))
		self.domain_discriminator.add_module('d_relu1', nn.ReLU())
		self.domain_discriminator.add_module('d_drop1', nn.Dropout(0.2))
		
		self.domain_discriminator.add_module('d_fc2', nn.Linear(512, 256))
		self.domain_discriminator.add_module('d_relu2', nn.ReLU())
		self.domain_discriminator.add_module('d_drop2', nn.Dropout(0.2))
		self.domain_discriminator.add_module('d_fc3', nn.Linear(256, 2))
		self.domain_discriminator.add_module('d_sfmax', nn.LogSoftmax(dim=1))
		#self.domain_discriminator.add_module('d_relu2', nn.ReLU())
		#self.domain_discriminator.add_module('d_drop2', nn.Dropout())
		#self.domain_discriminator.add_module('d_fc3', nn.Linear(1024, 1))

		self.optimizer = optimizer(list(self.domain_discriminator.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

		self.initialize_params()

		# TODO Check the RP size
		self.projection = nn.Linear(n_outputs, rp_size, bias=False)
		with torch.no_grad():
			self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

	def forward(self, input_data):
		#reverse_feature = ReverseLayer.apply(input_data, alpha)	# Make sure there will be no problem when updating discs params
		feature = input_data.view(input_data.size(0), -1)
		feature_proj = self.projection(feature)
		
		domain_output = self.domain_discriminator(feature_proj)

		return domain_output

	def initialize_params(self):

		for layer in self.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
			elif isinstance(layer, torch.nn.Linear):
				init.kaiming_uniform_(layer.weight)
			elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
