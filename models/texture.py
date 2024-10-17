import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp, get_encoding_with_network
from systems.utils import update_module_step
import os
from fractions import Fraction


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        self.encoding_with_network = get_encoding_with_network(self.n_dir_dims, self.n_output_dims, self.config.dir_encoding_config, self.config)

    def encoding(self, dirs):
        return self.encoding_with_network.encoding(dirs)

    def forward(self, features, dirs, group_id=None, group_id_dict=None, stage=None, img=None, *args):
        dirs = (dirs + 1.) / 2.  # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        z = self.get_z(img, group_id, group_id_dict, stage)
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        output = self.encoding_with_network.run_hyper_net(network_inp, z)
        color = output.view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def get_z(self, img, group_id, group_id_dict, stage):
        if self.encoding_with_network.latent_net is not None and img is not None:
            return self.encoding_with_network.latent_net(img)
        elif self.encoding_with_network.random_latent or group_id is None:
            return torch.randn((1, self.encoding_with_network.latent_dim)).cuda()
        else:
            return self.get_z_for_group_id(group_id, group_id_dict, stage)

    def get_z_for_group_id(self, group_id, group_id_dict, stage):
        if stage not in ['val', 'test']:
            return self.encoding_with_network.latent_net(group_id[0].long()).unsqueeze(0)
        else:
            if self.config.multi_joint_interpolation:
                return self.multi_joint_interpolation(group_id, group_id_dict)
            else:
                return self.interpolate_z(group_id, group_id_dict)

    def interpolate_z(self, group_id, group_id_dict):
        max_group_id = max(group_id_dict.keys())
        min_group_id = min(group_id_dict.keys())
        max_id = group_id_dict[max_group_id]
        min_id = group_id_dict[min_group_id]
        if group_id.long() == max_id:
            z1 = self.encoding_with_network.latent_net(group_id[0].long()).unsqueeze(0)
            second_id = torch.tensor([int(min_id)]).cuda()
            z2 = self.encoding_with_network.latent_net(second_id.long()).unsqueeze(0).squeeze(1)
            alpha = float(Fraction(self.config.factor))
            z = z1 * alpha + z2 * (1 - alpha)
        else:
            z = self.encoding_with_network.latent_net(group_id[0].long()).unsqueeze(0)
        return z

    def multi_joint_interpolation(self, group_id, group_id_dict):
        group_ids = list(group_id_dict.keys())
        group_ids.sort()
        joint_ids = self.separate_group_ids(group_ids)
        for joint_index, joint in enumerate(joint_ids):
            max_id, min_id = self.get_max_min_ids(joint)
            if group_id.long() == max_id:
                z1 = self.encoding_with_network.latent_net(group_id[0].long()).unsqueeze(0)
                second_id = torch.tensor([int(min_id)]).cuda()
                z2 = self.encoding_with_network.latent_net(second_id.long()).unsqueeze(0).squeeze(1)
                alpha = float(Fraction(self.config.factor))
                return z1 * alpha + z2 * (1 - alpha)
        return self.encoding_with_network.latent_net(group_id[0].long()).unsqueeze(0)

    def separate_group_ids(self, group_ids):
        joint1_ids, joint2_ids, joint3_ids = [], [], []
        for ele in group_ids:
            if ele < 40:
                joint1_ids.append(ele)
            elif ele < 80:
                joint2_ids.append(ele)
            else:
                joint3_ids.append(ele)
        return [joint1_ids, joint2_ids, joint3_ids]

    def get_max_min_ids(self, joint_ids):
        return max(joint_ids), min(joint_ids)

    def save_embeddings(self, group_id, z, stage, model_name='volume_radiance'):
        if stage not in ['test']:
            str_group_id = str(int(group_id[0].long()))
            save_dir = os.path.join('embeddings', self.config.data_dir)
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f'{str_group_id}_{model_name}_z.pt')
            torch.save(z, file_path)

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}

@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
