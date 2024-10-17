import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, SSIM, LPIPS, latent_manifold_loss, binary_cross_entropy, depth_smoothness_reg, occlusion_reg
from utils.chamfer import *
from utils import *
import random
import os
from PIL import Image
import cv2
import torchvision.transforms.functional as TF
from fractions import Fraction
from models.utils import cleanup

@systems.register('nerf-system')
class NeRFSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM(),
            'lpips': LPIPS(),
            'latent_manifold_loss': latent_manifold_loss(self.config),
            'depth_smoothness_reg': depth_smoothness_reg(self.config),
            'occlusion_reg': occlusion_reg(self.config)

        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays

    def forward(self, batch):     
        return self.model(batch['rays'], batch['group_id'], batch['group_id_dict'], batch['stage'])
    
    def preprocess_data(self, batch, stage):
        
        if self.config.model.batch_image_sampling:
            index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
        else:
            index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        
        # Get all unique group IDs from the dataset
        unique_group_ids = list(set(self.dataset.all_group_ids))
        # create a dict mapping unique_group_ids to 0,1,2,3 etc
        group_id_dict = {unique_group_ids[i]: i for i in range(len(unique_group_ids))}
        self.group_id_dict = group_id_dict
        # Select a random group ID from the dataset
        selected_group_id = random.choice(unique_group_ids)
        # Get the indices of all images in the dataset that belong to the selected group ID
        selected_indices = [i for i, x in enumerate(self.dataset.all_group_ids) if x == selected_group_id]
        # select indices from above that are of same size as self.train_num_rays using random.choices
        group_indices = random.choices(selected_indices, k=self.train_num_rays)
        group_indices.sort()
        group_indices = torch.tensor(group_indices, device=self.dataset.all_images.device)
        group_id_list = [group_id_dict[selected_group_id]] * self.train_num_rays
        index = group_indices


        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            index = index.to("cpu")
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            group_id = torch.Tensor(group_id_list).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        
        else:
            
            index = batch['index']
            group_id = group_id_dict[self.dataset.all_group_ids[index.item()]]
            group_id = torch.Tensor([group_id]).to(self.rank)
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            index = index.to("cpu")
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])        
    
        batch.update({
            'index': index,
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'group_id': group_id,
            'group_id_dict': group_id_dict,
            'stage': stage
        })
        cleanup()


    def get_interpolated_id(self, group_id, second_id):
        alpha = float(Fraction(self.config.model.geometry.factor))
        reverse_group_id_dict = dict(zip(self.group_id_dict.values(), self.group_id_dict.keys()))
        org_first_id = reverse_group_id_dict[group_id]
        org_second_id = reverse_group_id_dict[second_id]
        return org_first_id * alpha + org_second_id * (1 - alpha)

    def get_image_from_id(self, image_id, batch):
        original_image_name = self.dataset.all_image_names[batch['index'][0].item()]
        original_camera = original_image_name.split('_')[0].zfill(2)
        image_name = f"{original_camera}_{str(image_id).zfill(2)}.png"
        image_path = os.path.join(self.dataset.config.root_dir, 'images', image_name)
        image = Image.open(image_path)
        img_wh = self.dataset.img_wh
        image = image.resize(img_wh, Image.BICUBIC)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            new_img = Image.new('RGB', image.size, (255, 255, 255))
            new_img.paste(image, (0, 0), image)
            image = new_img
        else:
            image = image.convert('RGB')
        image = TF.to_tensor(image).permute(1, 2, 0)[..., :3]
        return image.view(-1, self.dataset.all_images.shape[-1]).to(self.rank)

    def get_metrics(self, out, batch, interpolated_image=None):
        W, H = self.dataset.img_wh
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        ssim_input = out['comp_rgb'].unsqueeze(0).reshape(1, 3, H, W).cpu()
        ssim_target = batch['rgb'].unsqueeze(0).reshape(1, 3, H, W).cpu()
        ssim = self.criterions['ssim'](ssim_input, ssim_target)
        lpips = self.criterions['lpips'](ssim_input, ssim_target)
        interpolated_metrics = {'psnr': torch.tensor(0.0), 'ssim': torch.tensor(0.0), 'lpips': torch.tensor(0.0)}
        if interpolated_image is not None:
            ssim_target_interpolated = interpolated_image.unsqueeze(0).reshape(1, 3, H, W).cpu()
            interpolated_metrics['psnr'] = self.criterions['psnr'](out['comp_rgb'].to(interpolated_image), interpolated_image)
            interpolated_metrics['ssim'] = self.criterions['ssim'](ssim_input, ssim_target_interpolated)
            interpolated_metrics['lpips'] = self.criterions['lpips'](ssim_input, ssim_target_interpolated)
        return psnr, ssim, lpips, interpolated_metrics    
    

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0
        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
        self.log('train/num_rays', float(self.train_num_rays))
        self.log( 'train/loss_rgb', loss_rgb)

        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)
         # object mask 
    
        loss_mask = (binary_cross_entropy(out['opacity'], batch['fg_mask'].float()))
        self.log('train/loss_mask', loss_mask, prog_bar=True)
        loss += loss_mask * self.config.system.loss.lambda_mask


        if self.C(self.config.model.geometry.latent_manifold):
            loss_latent_manifold = self.criterions['latent_manifold_loss'](int(batch['group_id'][0].item()), self.model.geometry.encoding_with_network.latent_net.weight)
            self.log('train/loss_latent_manifold', loss_latent_manifold)
            loss += loss_latent_manifold * self.C(self.config.system.loss.lambda_latent_manifold)
        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        if self.C(self.config.model.geometry.depth_reg):
            loss_depth_reg = self.criterions['depth_smoothness_reg'](out['depth'])
            self.log('train/loss_depth_reg', loss_depth_reg)
            loss += loss_depth_reg * self.C(self.config.system.loss.lambda_depth_smoothness)

        if self.C(self.config.model.geometry.occ_reg):
            loss_occlusion_reg = self.criterions['occlusion_reg'](out['density'])
            self.log('train/loss_occlusion_reg', loss_occlusion_reg)
            loss += loss_occlusion_reg * self.C(self.config.system.loss.lambda_occlusion_reg)
        
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        self.log('train/loss', loss)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        cleanup()
        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        int_group_id = int(batch['group_id'][0].item())
        max_group_id = max(self.group_id_dict.keys())
        min_group_id = min(self.group_id_dict.keys())
        max_id = self.group_id_dict[max_group_id]
        min_id = self.group_id_dict[min_group_id]
        self.max_id = max_id
        second_id = min_id
        interpolated_image = None
        if int_group_id == max_id and not self.config.dataset.real_images:
            interpolated_id = self.get_interpolated_id(int_group_id, second_id)
            if interpolated_id.is_integer():
                interpolated_id = int(interpolated_id)
                interpolated_image = self.get_image_from_id(interpolated_id, batch)
                reverse_group_id_dict = dict(zip(self.group_id_dict.values(), self.group_id_dict.keys()))
                actual_second_id = reverse_group_id_dict[second_id]
                second_image = self.get_image_from_id(actual_second_id, batch)
                images_to_save = [
                    {'type': 'rgb', 'img': batch['rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
                    {'type': 'rgb', 'img': second_image.view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
                    {'type': 'rgb', 'img': out['comp_rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}}
                ]
                if interpolated_image is not None:
                    images_to_save.insert(2, {'type': 'rgb', 'img': interpolated_image.view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}})
                self.save_image_grid(f"it{self.global_step}-interpolated-{batch['index'][0].item()}.png", images_to_save)
        
            psnr, ssim, lpips, interpolated_metrics = self.get_metrics(out, batch, interpolated_image)
        else:

            self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
            ])
            psnr, ssim, lpips, interpolated_metrics = self.get_metrics(out, batch)


        output = {
            'psnr': psnr,
            'interpolated_psnr': interpolated_metrics['psnr'],
            'ssim': ssim,
            'interpolated_ssim': interpolated_metrics['ssim'],
            'lpips': lpips,
            'interpolated_lpips': interpolated_metrics['lpips'],
            'index': batch['index'],
            'group_id': int(batch['group_id'][0])
        }
        self.validation_step_outputs.append(output)
        cleanup()
        return output

    def test_step(self, batch, batch_idx):
        out = self(batch)
        int_group_id = int(batch['group_id'][0].item())
        max_group_id = max(self.group_id_dict.keys())
        min_group_id = min(self.group_id_dict.keys())
        max_id = self.group_id_dict[max_group_id]
        min_id = self.group_id_dict[min_group_id]
        self.max_id = max_id

        if self.config.dataset.real_images:
            second_id = max(self.group_id_dict.keys() - {max_group_id})
            second_id = self.group_id_dict[second_id]
        else:
            second_id = min_id

        interpolated_image = None
        if int_group_id == max_id:
            if self.config.dataset.real_images:
                images_to_save = [
                    {'type': 'rgb', 'img': batch['rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
                    {'type': 'rgb', 'img': out['comp_rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}}
                ]
                if self.config.dataset.gradual_interpolation:
                    alpha = float(Fraction(self.config.model.geometry.factor))
                    imgs = self.get_image_grid_([images_to_save])
                    save_path = self.get_save_path(f"it{self.global_step}-test/{int(batch['group_id'][0])}/{batch['index'][0].item()}_{alpha}.png")
                    # save_path = os.path.join(save_path.split('@')[0], save_path.split('@')[1].split('/')[-1])
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    print("saving image at", save_path)
                    cv2.imwrite(save_path, imgs)
                    print("done saving image")
                # self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", images_to_save)
            else:
                interpolated_id = self.get_interpolated_id(int_group_id, second_id)
                if interpolated_id.is_integer():
                    interpolated_id = int(interpolated_id)
                    interpolated_image = self.get_image_from_id(interpolated_id, batch)
                    reverse_group_id_dict = dict(zip(self.group_id_dict.values(), self.group_id_dict.keys()))
                    actual_second_id = reverse_group_id_dict[second_id]
                    second_image = self.get_image_from_id(actual_second_id, batch)
                    images_to_save = [
                        {'type': 'rgb', 'img': batch['rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
                        {'type': 'rgb', 'img': second_image.view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}},
                        {'type': 'rgb', 'img': out['comp_rgb'].view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}}
                    ]
                    if interpolated_image is not None:
                        images_to_save.insert(2, {'type': 'rgb', 'img': interpolated_image.view(self.dataset.img_wh[1], self.dataset.img_wh[0], 3), 'kwargs': {'data_format': 'HWC'}})
                    if self.config.dataset.gradual_interpolation:
                        alpha = float(Fraction(self.config.model.geometry.factor))
                        imgs = self.get_image_grid_([images_to_save])
                        save_path = self.get_save_path(f"it{self.global_step}-test/{int(batch['group_id'][0])}/{batch['index'][0].item()}_{alpha}.png")
                        # save_path = os.path.join(save_path.split('@')[0], save_path.split('@')[1].split('/')[-1])
                        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        print("saving image at", save_path)
                        cv2.imwrite(save_path, imgs)
                        print("done saving image")
                        psnr, ssim, lpips= torch.Tensor([0.0]), torch.Tensor([0.0]), torch.Tensor([0.0])
                        interpolated_metrics = {'psnr': torch.Tensor([0.0]), 'ssim': torch.Tensor([0.0]), 'lpips': torch.Tensor([0.0])}
                    else:
                        psnr, ssim, lpips, interpolated_metrics = self.get_metrics(out, batch, interpolated_image)
                        self.save_image_grid(f"it{self.global_step}-test/interpolated_{int(batch['group_id'][0])}/{batch['index'][0].item()}.png", images_to_save)

        
        if self.config.dataset.real_images:
            psnr, ssim, lpips= torch.Tensor([0.0]), torch.Tensor([0.0]), torch.Tensor([0.0])
            interpolated_metrics = {'psnr': torch.Tensor([0.0]), 'ssim': torch.Tensor([0.0]), 'lpips': torch.Tensor([0.0])}
        if self.config.system.loss.eval_CD:
            self.export_meshes(batch['group_id'][0].item(), batch['index'][0].item(), self.group_id_dict, batch['stage'])
            cd_test = torch.Tensor([self.metrics_surface()])
        else:
            cd_test = torch.Tensor([0.0])
        if not int_group_id == max_id:
            psnr, ssim, lpips, interpolated_metrics = self.get_metrics(out, batch)

        output = {
            'psnr': psnr,
            'interpolated_psnr': interpolated_metrics['psnr'],
            'ssim': ssim,
            'interpolated_ssim': interpolated_metrics['ssim'],
            'lpips': lpips,
            'interpolated_lpips': interpolated_metrics['lpips'],
            'cd': cd_test,
            'index': batch['index'],
            'group_id': int(batch['group_id'][0])
        }
        self.test_step_outputs.append(output)
        return output
  
    def log_metrics_per_group(self, step_outputs, metric_name, log_prefix):
        metrics_per_group = {}
        for step_out in step_outputs:
            group_id = step_out['group_id']
            if group_id not in metrics_per_group:
                metrics_per_group[group_id] = []
            metrics_per_group[group_id].append(step_out[metric_name])

        mean_metrics_per_group = {group_id: torch.mean(torch.stack(metrics)) for group_id, metrics in metrics_per_group.items()}
        for group_id, mean_metric in mean_metrics_per_group.items():
            self.log(f'{log_prefix}/{metric_name}_group_{group_id}', mean_metric, prog_bar=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            for metric_name in ['psnr', 'ssim', 'lpips', 'interpolated_psnr', 'interpolated_ssim', 'interpolated_lpips']:
                self.log_metrics_per_group(self.validation_step_outputs, metric_name, 'val')
        cleanup()

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            for metric_name in ['interpolated_psnr', 'interpolated_ssim', 'interpolated_lpips', 'cd']:
                self.log_metrics_per_group(self.test_step_outputs, metric_name, 'test')

        cleanup()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )    
    def export_meshes(self, group_id, index, group_id_dict, stage):
        it = int(self.global_step)
        # configurations
        res = self.config.model.geometry.isosurface.resolution
        thre = float(self.config.model.geometry.isosurface.threshold)

        # extract geometry from the fields
        mesh_dict = self.model.isosurface(group_id, group_id_dict, stage)
        # save mesh in the canonical state
        self.save_mesh_ply(f"it{it}_static_{res}_thre{thre}.ply", **mesh_dict)
            
    def metrics_surface(self):
        it = int(self.global_step)
        # configurations
        res = self.config.model.geometry.isosurface.resolution
        thre = float(self.config.model.geometry.isosurface.threshold)

        # Chamfer-L1 Distance at start state
        cd_s= eval_CD(
            self.get_save_path(f"it{it}_static_{res}_thre{thre}.ply"),
            os.path.join((self.config.model.motion_gt_path), 'start', 'start_rotate.ply')
        )
        return cd_s