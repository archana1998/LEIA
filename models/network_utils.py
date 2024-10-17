import math
import numpy as np

import torch
import torch.nn as nn
import tinycudann as tcnn

from torch import Tensor
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info
from collections import OrderedDict
from utils.misc import config_to_primitive, get_rank
from models.utils import get_activation
from systems.utils import update_module_step
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
from . import hyper_net
from .layers import layer_utils
import os
from fractions import Fraction


class VanillaFrequency(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get('n_masking_step', 0)
        self.update_step(None, None) # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq*x) * mask]                
        return torch.cat(out, -1)          

    def update_step(self, epoch, global_step):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
        else:
            self.mask = (1. - torch.cos(math.pi * (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs)).clamp(0, 1))) / 2.
            rank_zero_debug(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')


class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.n_input_dims = in_channels
        encoding_config = config.copy()
        encoding_config['otype'] = 'HashGrid'
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = config['n_levels']
        self.n_features_per_level = config['n_features_per_level']
        self.start_level, self.start_step, self.update_steps = config['start_level'], config['start_step'], config['update_steps']
        self.current_level = self.start_level
        self.mask = torch.zeros(self.n_level * self.n_features_per_level, dtype=torch.float32, device=get_rank())

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask
        return enc

    def update_step(self, epoch, global_step):
        current_level = min(self.start_level + max(global_step - self.start_step, 0) // self.update_steps, self.n_level)
        if current_level > self.current_level:
            rank_zero_info(f'Update grid level to {current_level}')
        self.current_level = current_level
        self.mask[:self.current_level * self.n_features_per_level] = 1.


class CompositeEncoding(nn.Module):
    def __init__(self, encoding, include_xyz=False, xyz_scale=1., xyz_offset=0.):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = int(self.include_xyz) * self.encoding.n_input_dims + self.encoding.n_output_dims
    
    def forward(self, x, *args):
        return self.encoding(x, *args) if not self.include_xyz else torch.cat([x * self.xyz_scale + self.xyz_offset, self.encoding(x, *args)], dim=-1)

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)


def get_encoding(n_input_dims, config):
    # input suppose to be range [0, 1]
    if config.otype == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, config_to_primitive(config))
    elif config.otype == 'ProgressiveBandHashGrid':
        encoding = ProgressiveBandHashGrid(n_input_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            encoding = tcnn.Encoding(n_input_dims, config_to_primitive(config))
    encoding = CompositeEncoding(encoding, include_xyz=config.get('include_xyz', False), xyz_scale=2., xyz_offset=-1.)
    return encoding


class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])
    
    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


def fmm_modulate_linear(og_weight: Tensor, modulation: Tensor, activation: str = "demod") -> Tensor:
    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = og_weight * (modulation + 1.0)  # [c_out, c_in]

    if activation == "demod":
        W = W / (W.norm(dim=2, keepdim=True) + 1e-8)  # [batch_size, c_out, c_in]
    return W


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    # weirdly all args should be before params.
    def forward(self, input, params=None, **kwargs):
        f"""
			Args: 
				input: input for the layer. 
				params: OrderedDict of parameters from hypernet.
				params can be empty when we skip a certain layer in hypernet.
		"""

        if params is None or params == {}:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        if 'hypernet_output_type' in kwargs and weight.ndim > 2:

            """
                Should only be applied to weights where needed. 
            """

            hypernet_output_type = kwargs['hypernet_output_type']
            if hypernet_output_type == 'soft_mask':
                og_weight = self.weight
                # assert the matrices arent the same.
                assert not torch.equal(og_weight, weight)

                modulation = weight  # we get the modulation from hypernet.
                weight = fmm_modulate_linear(og_weight, modulation, activation=kwargs['activation'])
                weight = weight.to(input.dtype)

        weight = weight.to(torch.float16)
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)

        return output


class BatchVanillaMLP(MetaModule):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = []
        self.layers.append(MetaSequential(self.make_batch_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()))
        for i in range(self.n_hidden_layers - 1):
            self.layers.append(MetaSequential(self.make_batch_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False),
                                              self.make_activation()))
        self.layers.append(MetaSequential(self.make_batch_linear(self.n_neurons, dim_out, is_first=False, is_last=True)))
        self.layers = MetaSequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])

    # @torch.cuda.amp.autocast(False)
    def forward(self, inputs, hyper=False, params=None, **kwargs):
        """
            Even hypernet is built using FCblock.
            hyper flag is used to differentiate between the two, for debugging purposes.
        """
        if params is None or params == {}:
            params = OrderedDict(self.named_parameters())

        output = self.layers(inputs, params=get_subdict(params, 'layers'), **kwargs)
        return output

    def make_batch_linear(self, dim_in, dim_out, is_first, is_last):
        layer = BatchLinear(dim_in, dim_out)  # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)

def sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network):
    rank_zero_debug('Initialize tcnn MLP to approximately represent a sphere.')
    """
    from https://github.com/NVlabs/tiny-cuda-nn/issues/96
    It's the weight matrices of each layer laid out in row-major order and then concatenated.
    Notably: inputs and output dimensions are padded to multiples of 8 (CutlassMLP) or 16 (FullyFusedMLP).
    The padded input dimensions get a constant value of 1.0,
    whereas the padded output dimensions are simply ignored,
    so the weights pertaining to those can have any value.
    """
    padto = 16 if config.otype == 'FullyFusedMLP' else 8
    n_input_dims = n_input_dims + (padto - n_input_dims % padto) % padto
    n_output_dims = n_output_dims + (padto - n_output_dims % padto) % padto
    data = list(network.parameters())[0].data
    assert data.shape[0] == (n_input_dims + n_output_dims) * config.n_neurons + (config.n_hidden_layers - 1) * config.n_neurons**2
    new_data = []
    # first layer
    weight = torch.zeros((config.n_neurons, n_input_dims)).to(data)
    torch.nn.init.constant_(weight[:, 3:], 0.0)
    torch.nn.init.normal_(weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
    new_data.append(weight.flatten())
    # hidden layers
    for i in range(config.n_hidden_layers - 1):
        weight = torch.zeros((config.n_neurons, config.n_neurons)).to(data)
        torch.nn.init.normal_(weight, 0.0, math.sqrt(2) / math.sqrt(config.n_neurons))
        new_data.append(weight.flatten())
    # last layer
    weight = torch.zeros((n_output_dims, config.n_neurons)).to(data)
    torch.nn.init.normal_(weight, mean=math.sqrt(math.pi) / math.sqrt(config.n_neurons), std=0.0001)
    new_data.append(weight.flatten())
    new_data = torch.cat(new_data)
    data.copy_(new_data)


def get_mlp(n_input_dims, n_output_dims, config):
    if config.otype == 'VanillaMLP':
        network = VanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    elif config.otype == 'BatchVanillaMLP':
        network = BatchVanillaMLP(n_input_dims, n_output_dims, config_to_primitive(config))
    else:
        with torch.cuda.device(get_rank()):
            network = tcnn.Network(n_input_dims, n_output_dims, config_to_primitive(config))
            if config.get('sphere_init', False):
                sphere_init_tcnn_network(n_input_dims, n_output_dims, config, network)
    return network


class EncodingWithNetwork(nn.Module):
    def __init__(self, encoding, fc_block, latentnet, hypernet, hyper_net_args, time_pos_encoding, dim_out, latent_dim, network_config):
        super().__init__()
        self.encoding, self.fc_block, self.hyper_net, self.latent_net = encoding, fc_block, hypernet, latentnet
        self.time_pos_encoding = time_pos_encoding
        self.dim_out = dim_out
        self.hyper_net_args = hyper_net_args
        self.random_latent = False
        self.group_size = 1 # TODO: remove hardcoding
        self.latent_dim = latent_dim
        self.config = network_config

    def run_hyper_net(self, x, z, interpolated_id=None):            
        params = self.hyper_net(z) #weights of the fc_block
        out = self.fc_block(x, params=params, **self.hyper_net_args)
        out = out.view(out.shape[0], self.group_size, out.shape[1], self.dim_out)
        return out
    
    def get_z_from_img(self, img):
        return self.latent_net(img)

    def get_random_z(self):
        return torch.randn((1, self.latent_dim)).cuda()

    def get_z_from_group_id(self, group_id):
        return self.latent_net(group_id.long()).unsqueeze(0)

    def interpolate_z(self, group_id, group_id_dict):
        max_group_id = max(group_id_dict.keys())
        min_group_id = min(group_id_dict.keys())
        max_id = group_id_dict[max_group_id]
        min_id = group_id_dict[min_group_id]
        if group_id.long() == max_id:
            z1 = self.get_z_from_group_id(group_id)
            second_id = torch.tensor([int(min_id)]).cuda()
            z2 = self.get_z_from_group_id(second_id).squeeze(1)
            alpha = float(Fraction(self.config.factor))
            z = z1 * alpha + z2 * (1 - alpha)
        else:
            z = self.get_z_from_group_id(group_id)
        return z

    def separate_group_ids(self, group_ids):
        joint1_ids, joint2_ids, joint3_ids = [], [], []
        for ele in group_ids:
            if ele < 40:
                joint1_ids.append(ele)
            elif ele < 80:
                joint2_ids.append(ele)
            else:
                joint3_ids.append(ele)
        return joint1_ids, joint2_ids, joint3_ids

    def get_max_min_ids(self, joint_ids):
        return max(joint_ids), min(joint_ids)

    def multi_joint_interpolation(self, group_id, group_id_dict):
        group_ids = list(group_id_dict.keys())
        group_ids.sort()
        joint1_ids, joint2_ids, joint3_ids = self.separate_group_ids(group_ids)
        max_joint_1_id, min_joint_1_id = self.get_max_min_ids(joint1_ids)
        max_joint_2_id, min_joint_2_id = self.get_max_min_ids(joint2_ids)
        max_joint_3_id, min_joint_3_id = self.get_max_min_ids(joint3_ids)
        joint_max_min_ids = [
            (max_joint_1_id, min_joint_1_id),
            (max_joint_2_id, min_joint_2_id),
            (max_joint_3_id, min_joint_3_id),
        ]
        for max_id, min_id in joint_max_min_ids:
            if group_id.long() == max_id:
                z1 = self.get_z_from_group_id(group_id)
                second_id = torch.tensor([int(min_id)]).cuda()
                z2 = self.get_z_from_group_id(second_id).squeeze(1)
                alpha = float(Fraction(self.config.factor))
                return z1 * alpha + z2 * (1 - alpha)
        return self.get_z_from_group_id(group_id)

    def save_embeddings(self, group_id, z, group_id_dict, model_name='geometry'):
        if not self.config.multi_joint_interpolation:
            str_group_id = str(int(group_id[0].long()))
            save_dir = 'embeddings' + '/' + self.config.data_dir
            os.makedirs(save_dir, exist_ok=True)
            torch.save(z, os.path.join(save_dir,str_group_id + model_name + '_z.pt'))
        else:
            reverse_group_id_dict = dict(zip(group_id_dict.values(), group_id_dict.keys()))
            str_group_id = str(reverse_group_id_dict[int(group_id[0].long())])
            save_dir = 'embeddings' + '/' + self.config.data_dir
            os.makedirs(save_dir, exist_ok=True)
            torch.save(z, os.path.join(save_dir,str_group_id + model_name + '_z.pt'))

    def forward(self, coords, group_id=None, group_id_dict=None, stage=None, img=None, z=None, **kwargs):
        coords = self.encoding(coords)
        # breakpoint()
        if z is None:
            if self.latent_net is not None and img is not None:
                z = self.get_z_from_img(img)
            elif self.random_latent or group_id is None:
                z = self.get_random_z()
            else:
                # adding interpolation in val state
                if stage in ['train']:
                    # breakpoint()
                    z = self.get_z_from_group_id(group_id[0])
                else:
                    # breakpoint()
                    if not self.config.multi_joint_interpolation:
                        # breakpoint()
                        z = self.interpolate_z(group_id[0], group_id_dict)
                    else:
                        # breakpoint()
                        z = self.multi_joint_interpolation(group_id[0], group_id_dict)
                # breakpoint()
                if self.time_pos_encoding is not None:
                    z_t = self.time_pos_encoding(group_id[0].long())
                    z = z + z_t

        if group_id is not None and self.config.save_embeddings and stage not in ['test']:
            self.save_embeddings(group_id, z,  group_id_dict)        
        outputs = self.run_hyper_net(coords, z)
        return outputs
    
    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)
        update_module_step(self.fc_block, epoch, global_step)


def get_encoding_with_network(n_input_dims, n_output_dims, encoding_config, network_config):
    # input suppose to be range [0, 1]
    if encoding_config.otype in ['VanillaFrequency', 'ProgressiveBandHashGrid', ] \
        or network_config.mlp_network_config.otype in ['VanillaMLP', 'BatchVanillaMLP']:

        latent_dim = network_config.mlp_network_config.n_neurons
        latent_net = None
        encoding = get_encoding(n_input_dims, encoding_config)
        # checking for texture volume radiance model, input dims for get_mlp is different (check texture.py)
        if encoding_config.otype in ["SphericalHarmonics"]:
            encoding.n_output_dims = encoding.n_output_dims + network_config.input_feature_dim
        fc_block = get_mlp(encoding.n_output_dims, n_output_dims, network_config.mlp_network_config)

        ########## Setup Latent Network ##########
        latent_net_config = network_config.latent_network
        latent_dim = latent_net_config.dim
        if latent_net_config.type == 'train_latents':
            # TODO: update num frames in data loader, then following code should be used
            num_states = network_config.num_states #hardcode as that's many cameras we have            
            # Each Frame group gets its own latent code.
            group_size = 1
            num_latents = num_states // 1  \
                if num_states != 1 else num_states
            
            print('Using train latents. Number of latents: ', num_latents)
            latent_net = nn.Embedding(num_latents, latent_dim)
            nn.init.normal_(latent_net.weight, mean=0, std=0.01)  # can play with std.
        else:
            latent_net = nn.Identity()

        ######## Hyper Network ########################

        if network_config.hyper_net.layers is not None:
            hyper_layers = network_config.hyper_net.layers
        else:
            # include nerf layers
            hyper_layers = list(range(network_config.mlp_network_config.n_hidden_layers))

        hyper_net_args = {'hypernet_output_type': network_config.hyper_net.output_type,
                               'activation': network_config.hyper_net.mask_act}

        hyp_net = hyper_net.HyperNetwork(cfg=network_config, hyper_in_features=latent_dim, \
                                                hypo_modules=fc_block, hyper_layers=hyper_layers)
        # hypernet_output_type = network_config.hyper_net.output_type
        if latent_net_config.time_pos_encodings:
            time_pos_encoding = layer_utils.TimePosEncoding(cfg=network_config, dim=latent_dim, num_timesteps=num_states, freq=100)
        else:
            time_pos_encoding = None
        encoding_with_network = EncodingWithNetwork(encoding, fc_block, latent_net, hyp_net, hyper_net_args,
                                                    time_pos_encoding, n_output_dims, latent_dim, network_config)
    else:
        with torch.cuda.device(get_rank()):
            encoding_with_network = tcnn.NetworkWithInputEncoding(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                encoding_config=config_to_primitive(encoding_config),
                network_config=config_to_primitive(network_config)
            )
    return encoding_with_network
