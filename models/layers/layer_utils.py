import torch
import torch.nn as nn

import math 
import numpy as np


# class TimePosEncoding(nn.Module):
# 	def __init__(self, dim, num_timesteps, freq=100):
# 		super().__init__()
# 		assert dim > 1, "Embedding dimension must be greater than 1"
# 		# Create inverse frequency vector
# 		inv_freq = torch.zeros(dim)
# 		inv_freq[0::2] = torch.pow(1.0 / freq, torch.arange(0, dim - dim // 2, dtype=torch.float))
# 		inv_freq[1::2] = torch.pow(1.0 / freq, torch.arange(0, dim // 2, dtype=torch.float))

# 		# Generate position vector for timesteps
# 		pos_vec = inv_freq.unsqueeze(1) * torch.arange(num_timesteps, dtype=torch.float).unsqueeze(0)
# 		pos_vec[1::2, :] += torch.pi / 2

# 		# Create positional encoding
# 		self.pos_encoding = torch.sin(pos_vec).T

# 	def forward(self, group_idx):
# 		self.pos_encoding = self.pos_encoding.to(group_idx.device)
# 		return self.pos_encoding[group_idx]

class TimePosEncoding(nn.Module):
	def __init__(self, dim, num_timesteps,cfg,freq=100.0):
		super(TimePosEncoding, self).__init__()
		self.D = dim
		self.N = num_timesteps
		self.freq = freq
		self.cfg = cfg

		# if self.cfg.data.frame_stride is not None:
		# Does not work. 
		# 	self.N = (self.N * self.cfg.data.frame_stride) + 1

		# Create a 2D position encoding matrix with shape (N, D)
		position = torch.arange(self.N, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, self.D, 2).float() * (-math.log(self.freq) / self.D))
		self.pe = torch.zeros(self.N, self.D)
		self.pe[:, 0::2] = torch.sin(position * div_term)
		self.pe[:, 1::2] = torch.cos(position * div_term)

		# if self.cfg.data.frame_stride is not None:
		# 	#sample every frame_stride frames - corresponding time encoding.
		# 	self.pe = self.pe[::self.cfg.data.frame_stride,:]

		# Register pe as a buffer that is not a model parameter
		self.register_buffer('weight', self.pe)

	def forward(self, group_idx):
		# indices is a batch of indices of shape (B, 1)
		# Gather the positional encodings corresponding to these indices
		positional_encodings = self.weight[group_idx, :]
		positional_encodings = positional_encodings.to(group_idx.device)
		return positional_encodings

# class TimePosEncoding(nn.Module):
#     def __init__(self, dim, num_timesteps, cfg, freq=100.0):
#         super(TimePosEncoding, self).__init__()
#         self.D = dim
#         self.N = num_timesteps
#         self.freq = freq
#         self.cfg = cfg

#         # Create a 2D position encoding matrix with shape (N, D)
#         position = torch.arange(self.N, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.D, 2).float() * (-math.log(self.freq) / self.D))
#         self.pe = torch.zeros(self.N, self.D)
#         self.pe[:, 0::2] = torch.sin(position * div_term)
#         self.pe[:, 1::2] = torch.cos(position * div_term)

#         # Register pe as a buffer that is not a model parameter
#         self.register_buffer('weight', self.pe)

#     def forward(self, group_idx, t, T):
#         # Apply frequency regularization to the positional encodings
#         alpha = self.calculate_frequency_mask(t, T, self.D // 2)  # D // 2 because D includes sin and cos components
#         freq_reg_pe = self.weight * alpha
#         # Gather the positional encodings corresponding to these indices
#         positional_encodings = freq_reg_pe[group_idx, :]
#         positional_encodings = positional_encodings.to(group_idx.device)
#         return positional_encodings

#     def calculate_frequency_mask(self, t, T, L):
#         alpha = torch.zeros(L + 3)
#         # Define the three regions for the alpha mask
#         region1 = int(t * L / T) + 3
#         region2_start = region1 + 1
#         region2_end = int(t * L / T) + 6
#         region3_start = region2_end + 1
        
#         # Set the mask values for each region
#         alpha[:region1] = 1
#         alpha[region2_start:region2_end] = torch.linspace(start=(t * L / T), end=1 - (t * L / T), steps=region2_end - region2_start)
#         # The rest of the mask is already initialized to zero
        
#         return alpha.view(1, -1)  # Reshape to (1, L) for broadcasting during multiplication



class PosEncoding(nn.Module):
	"""
		Positional encoding used in SIREN+nerv block.
		Taken from NIRVANA. 
		Repeats frame G times and adds pos encoding.
	"""

	def __init__(self, dim, num_frames, freq):
		super().__init__()
		assert dim>1
		self.dim = dim
		self.num_frames = num_frames
		inv_freq = torch.zeros(dim)
		inv_freq[0::2] = torch.pow(1/freq, torch.arange(dim-dim//2))
		inv_freq[1::2] = torch.pow(1/freq, torch.arange(dim//2))
		pos_vec = inv_freq.unsqueeze(1)*torch.arange(num_frames).unsqueeze(0)
		pos_vec[1::2,:] += torch.pi/2
		#self.pos_encoding = nn.Parameter(torch.sin(pos_vec).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),requires_grad=False)
		self.pos_encoding = nn.Parameter(torch.sin(pos_vec).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),requires_grad=False)
		self.num_frames = num_frames
		#assert self.pos_encoding.size() == (1,dim,num_frames,1,1,1)
		assert self.pos_encoding.size() == (1,dim,num_frames,1,1)

	def forward(self, x):
		
		B, N, C, H, W = x.size()
		out = x.unsqueeze(3)  + self.pos_encoding #encoding

		return out.reshape(B,N, C*self.num_frames, H, W)

		# assert x.dim() == 4
		# N, C, H, W = x.size()
		# out = x.unsqueeze(2)+self.pos_encoding
		# return out.reshape(N, C*self.num_frames, H, W)

class Reshape_op(torch.nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape
		assert len(shape) == 3
		
	def forward(self, x):
		if x.ndim == 3:
			bs,num_features,feat_size = x.shape
		elif x.ndim == 2:
			num_features,feat_size = x.shape
			bs = 1
			
		x = x.view(bs,num_features,self.shape[0],self.shape[1],self.shape[2])
		return x


class Sine(nn.Module):
	"""Sine activation with scaling.

	Args:
		w0 (float): Omega_0 parameter from SIREN paper.
	"""
	def __init__(self, w0=1.):
		super().__init__()
		self.w0 = w0

	def forward(self, x):
		return torch.sin(self.w0 * x)


def get_activation(activation):
	
	if (activation == 'none') or (activation == 'linear') or (activation is None):
		return nn.Identity()

	elif activation.lower() == 'relu':
		return nn.ReLU()
	elif activation.lower() == 'leakyrelu':
		return nn.LeakyReLU()
	elif activation.lower() == 'tanh':
		return nn.Tanh()
	elif activation.lower() == 'sigmoid':
		return nn.Sigmoid()
	else:
		raise ValueError('Unknown activation function {}'.format(activation))



############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1e1

	# if hasattr(m, 'bias') and siren:
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m, siren=False):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
		m.weight.data = m.weight.data / 1.e1

	# if hasattr(m, 'bias') and siren:
	#     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
	#     with torch.no_grad():
	#         m.bias.uniform_(-1/fan_in, 1/fan_in)


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	# grab from upstream pytorch branch and paste here for now
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor


def init_weights_trunc_normal(m):
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	#if type(m) == BatchLinear or type(m) == nn.Linear:
	if hasattr(m, 'weight'):
		fan_in = m.weight.size(1)
		fan_out = m.weight.size(0)
		std = math.sqrt(2.0 / float(fan_in + fan_out))
		mean = 0.
		# initialize with the same behavior as tf.truncated_normal
		# "The generated values follow a normal distribution with specified mean and
		# standard deviation, except that values whose magnitude is more than 2
		# standard deviations from the mean are dropped and re-picked."
		_no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
	if hasattr(m, 'weight'):
		nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):	
	if hasattr(m, 'weight'):
		num_input = m.weight.size(-1)
		nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
	#if type(m) == BatchLinear or type(m) == nn.Linear:
	if hasattr(m, 'weight'):
		num_input = m.weight.size(-1)
		nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
	#if type(m) == BatchLinear or type(m) == nn.Linear:
	if hasattr(m, 'weight'):
		nn.init.xavier_normal_(m.weight)


def sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
	with torch.no_grad():
		if hasattr(m, 'weight'):
			num_input = m.weight.size(-1)
			# See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
			m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
	y = x.clone()
	y[..., 1::2] = -1 * y[..., 1::2]
	return y


def compl_div(x, y):
	''' x / y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = (a * c + b * d) / (c ** 2 + d ** 2)
	outi = (b * c - a * d) / (c ** 2 + d ** 2)
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out


def compl_mul(x, y):
	'''  x * y '''
	a = x[..., ::2]
	b = x[..., 1::2]
	c = y[..., ::2]
	d = y[..., 1::2]

	outr = a * c - b * d
	outi = (a + b) * (c + d) - a * c - b * d
	out = torch.zeros_like(x)
	out[..., ::2] = outr
	out[..., 1::2] = outi
	return out
