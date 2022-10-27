import torch
import torch.nn as nn

from .wgan_qc import WassersteinGanQuadraticCost
from .resnet_1 import ResNet_D, ResNet_G


def create_wgan(parameters, device, optimizer='adam'):
    if parameters['model'] == 'resnet':
        generator, discriminator = init_resnet(parameters)
    else:
        raise NotImplementedError

    if optimizer == 'adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'])
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'])
    elif optimizer == 'rmsprop':
        optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=parameters['learning_rate'])
        optimizer_d = torch.optim.RMSprop(generator.parameters(), lr=parameters['learning_rate'])

    criterion = torch.nn.MSELoss()

    gan = WassersteinGanQuadraticCost(generator,
                                      discriminator,
                                      optimizer_g,
                                      optimizer_d,
                                      criterion=criterion,
                                      data_dimensions=parameters['data_dim'],
                                      epochs=parameters['epochs'],
                                      batch_size=parameters['batch_size'],
                                      device=device,
                                      n_max_iterations=parameters['n_max_iterations'],
                                      gamma=parameters['gamma'])

    return gan


def init_resnet(parameters):
    critic = ResNet_D(parameters['data_dim'][-1], parameters['size'], nfilter=parameters['nfilter'],
                      nfilter_max=parameters['nfilter_max'])
    generator = ResNet_G(parameters['data_dim'][-1], parameters['z_dim'], parameters['size'],
                         nfilter=parameters['nfilter'], nfilter_max=parameters['nfilter_max'])

    generator.apply(weights_init_G)
    critic.apply(weights_init_D)

    return generator, critic


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
