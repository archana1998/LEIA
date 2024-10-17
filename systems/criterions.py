import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from skimage.metrics import structural_similarity as ssim
import numpy as np


class WeightedLoss(nn.Module):
    @property
    def func(self):
        raise NotImplementedError

    def forward(self, inputs, targets, weight=None, reduction='mean'):
        assert reduction in ['none', 'sum', 'mean', 'valid_mean']
        loss = self.func(inputs, targets, reduction='none')
        if weight is not None:
            while weight.ndim < inputs.ndim:
                weight = weight[..., None]
            loss *= weight.float()
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'valid_mean':
            return loss.sum() / weight.float().sum()


class MSELoss(WeightedLoss):
    @property
    def func(self):
        return F.mse_loss


class L1Loss(WeightedLoss):
    @property
    def func(self):
        return F.l1_loss

class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        lpips_model = lpips.LPIPS(net='vgg')
        value = lpips_model(inputs, targets)
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return torch.mean(value)
        elif reduction == 'none':
            return value





class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian
        
        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        
        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale)**2
        self.c2 = (k2 * data_scale)**2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):


        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = F.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = F.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0) : (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        elif reduction == 'sum':
            return _ssim.sum()
        elif reduction == 'mean':
            return _ssim.mean()

        
def binary_cross_entropy(input, target, clips_eps=1e-6):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    input = torch.clamp(input, clips_eps, 1. - clips_eps)
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

class latent_manifold_loss(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(latent_manifold_loss, self).__init__()
        self.loss = {}
        self.cfg = cfg
        self.neighbors = self.cfg.system.loss.latent_manifold_neighbors

    def forward(self, group_id, all_latents,**kwargs):
        group_id = group_id
        all_latents = all_latents
        selected_latents = all_latents[group_id]
        #diff bet each selected latent and all other latents
        diff = selected_latents[None,:] - all_latents 
        dist = torch.norm(diff, p=2, dim=0)
        dist_mean = dist.mean(dim=0)
        dist_small = torch.topk(dist, self.neighbors, dim=0, largest=False)[0]        
        # dist_small = dist_small[1:, :]
        self.loss = dist_small.mean()
        return self.loss
    
class depth_smoothness_reg(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(depth_smoothness_reg, self).__init__()
        self.loss = {}
        self.cfg = cfg
        # breakpoint()
        self.patch_size = self.cfg.system.loss.patch_size
    def forward(self, depths, **kwargs):
        # Calculate the smoothness loss for each ray
        # breakpoint()
        loss = 0
        if len(depths[0])==1:
            d_r = depths
            for i in range(self.patch_size-1):
                # Add smoothness loss for each adjacent pixel in the patch
                loss += (d_r[i] - d_r[i + 1]) ** 2 + (d_r[i] - d_r[(i + 1) % self.patch_size]) ** 2
        else:
            for r in range(len(depths)):
                d_r = depths[r]
                # if len(d_r)==1:
                #     d_r = depths
                for i in range(self.patch_size-1):
                    # Add smoothness loss for each adjacent pixel in the patch
                    loss += (d_r[i] - d_r[i + 1]) ** 2 + (d_r[i] - d_r[(i + 1) % self.patch_size]) ** 2

        # breakpoint()
        mean_loss = loss.item() / len(depths)
        return mean_loss
    
class frequency_reg(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(frequency_reg, self).__init__()
        self.loss = {}
        self.cfg = cfg
        self.L = self.cfg.system.loss.L
        self.T = self.cfg.system.loss.T
        
    def calculate_frequency_mask(t, T, L):
        alpha = torch.zeros(L + 3)
        # Define the three regions for the alpha mask
        region1 = int(t * L / T) + 3
        region2_start = region1 + 1
        region2_end = int(t * L / T) + 6
        # region3_start = region2_end + 1
        
        # Set the mask values for each region
        alpha[:region1] = 1
        alpha[region2_start:region2_end] = torch.linspace(start=(t * L / T), end=1 - (t * L / T), steps=region2_end - region2_start)
        # The rest of the mask is already initialized to zero
        
        return alpha

    def apply_frequency_regularization(gamma_L, t, T, L):
        # Calculate the frequency mask
        alpha = calculate_frequency_mask(t, T, L)
        # Apply the mask to the positional encoding element-wise
        gamma_t = gamma_L * alpha
        return gamma_t

class occlusion_reg(nn.Module):
    def __init__(self,cfg,**kwargs):
        super(occlusion_reg, self).__init__()
        self.loss = {}
        self.cfg = cfg
        self.regularization_range = self.cfg.system.loss.regularization_range
    def forward(self, density_values, **kwargs):
        # breakpoint()
        # Ensure the binary mask is set to 1 up to the regularization range and 0 afterwards
        mask = torch.zeros_like(density_values)
        mask[:, :self.regularization_range] = 1

        # Compute the occlusion loss as the average of the product of the binary mask and density values
        occlusion_loss = (density_values * mask).sum(dim=1).mean()
        return occlusion_loss


# def occlusion_regularization_loss(density_values, binary_mask, regularization_range):
#     """
#     Compute the occlusion regularization loss.

#     Parameters:
#     density_values (torch.Tensor): A tensor of shape (N, K) where N is the number of rays,
#                                    and K is the number of sampled points along each ray.
#                                    It contains the density values (sigma) for each sampled point.
#     binary_mask (torch.Tensor): A tensor of the same shape as density_values, containing binary
#                                 values that determine whether a point will be penalized.
#     regularization_range (int): The index M up to which the values of the binary mask are set to 1,
#                                 and the rest to 0.

#     Returns:
#     torch.Tensor: The computed occlusion regularization loss.
#     """
#     # Ensure the binary mask is set to 1 up to the regularization range and 0 afterwards
#     mask = torch.zeros_like(binary_mask)
#     mask[:, :regularization_range] = 1

#     # Compute the occlusion loss as the average of the product of the binary mask and density values
#     occlusion_loss = (density_values * mask).sum(dim=1).mean()

#     return occlusion_loss

# # Example usage:
# # Assuming N rays, K sampled points along each ray, and M as the regularization range
# N, K, M = 512, 100, 10  # Example values for demonstration
# density_values = torch.rand(N, K)  # Random density values for each sampled point
# binary_mask = torch.ones(N, K)  # Binary mask with all ones for simplicity

# # Compute the occlusion regularization loss
# loss_occ = occlusion_regularization_loss(density_values, binary_mask, M)
