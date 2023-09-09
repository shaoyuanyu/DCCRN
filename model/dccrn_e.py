import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d

from .conv_stft import ConvSTFT, ConviSTFT
from .complexnn import ComplexConv2d, ComplexConvTranspose2d, complex_cat

def remove_dc(data):
	mean = torch.mean(data, -1, keepdim=True) 
	data = data - mean
	return data

def l2_norm(s1, s2):
	norm = torch.sum(s1*s2, -1, keepdim=True)
	return norm

def si_snr(s1, s2, eps=1e-8):
	s1_s2_norm = l2_norm(s1, s2)
	s2_s2_norm = l2_norm(s2, s2)
	s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
	e_nosie = s1 - s_target
	target_norm = l2_norm(s_target, s_target)
	noise_norm = l2_norm(e_nosie, e_nosie)
	snr = 10 * torch.log10( target_norm / (noise_norm + eps) + eps )
	return torch.mean(snr)

class DCCRN(nn.Module):
	def __init__(self):
		super(DCCRN, self).__init__()

		self.rnn_layers = 2
		self.win_len = 400
		self.win_inc = 100
		self.fft_len = 512
		self.rnn_units = 128
		self.input_dim = self.win_len
		self.output_dim = self.win_len
		self.hidden_layers = self.rnn_layers
		self.kernel_size = 5
		self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
		self.fac = 1
		self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, 'hanning', 'complex', fix=True)
		self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, 'hanning', 'complex', fix=True)
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()

		for idx in range(len(self.kernel_num) - 1):
			self.encoder.append(
				nn.Sequential(
					ComplexConv2d(
						self.kernel_num[idx],
						self.kernel_num[idx+1],
						kernel_size = (self.kernel_size, 2),
						stride = (2, 1),
						padding = (2, 1)
					),
					nn.BatchNorm2d(self.kernel_num[idx+1]),
					nn.PReLU()
				)
			)
		hidden_dim = self.fft_len // ( 2 ** len(self.kernel_num) ) 

		self.enhance = nn.LSTM(
				input_size = hidden_dim * self.kernel_num[-1],
				hidden_size = self.rnn_units,
				num_layers = 2,
				dropout = 0.0,
				bidirectional = False,
				batch_first = False
		)
		self.tranform = nn.Linear(self.rnn_units * self.fac, hidden_dim * self.kernel_num[-1])

		for idx in range(len(self.kernel_num)-1, 0, -1):
			if idx != 1:
				self.decoder.append(
					nn.Sequential(
						ComplexConvTranspose2d(
						self.kernel_num[idx] * 2,
						self.kernel_num[idx-1],
						kernel_size = (self.kernel_size, 2),
						stride = (2, 1),
						padding = (2, 0),
						output_padding = (1, 0)
						),
					nn.BatchNorm2d(self.kernel_num[idx-1]),
					#nn.ELU()
					nn.PReLU()
					)
				)
			else:
				self.decoder.append(
					nn.Sequential(
						ComplexConvTranspose2d(
						self.kernel_num[idx] * 2,
						self.kernel_num[idx-1],
						kernel_size = (self.kernel_size, 2),
						stride = (2, 1),
						padding = (2, 0),
						output_padding = (1, 0)
						),
					)
				)
		
		#show_model(self)
		#show_params(self)
		self.flatten_parameters() 

	def flatten_parameters(self): 
		if isinstance(self.enhance, nn.LSTM):
			self.enhance.flatten_parameters()

	def forward(self, inputs, lens=None):
		specs = self.stft(inputs)
		real = specs[:, :(self.fft_len // 2 + 1)]
		imag = specs[:, (self.fft_len // 2 + 1):]

		del specs

		spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

		spec_phase = torch.atan2(imag, real)

		cspecs = torch.stack([real, imag], 1)
		out = cspecs[:, :, 1:]
		out0 = out.clone()

		del real, imag, cspecs

		#encoder_out = []
		
		for layer in self.encoder:
			out = layer(out)
			#encoder_out.append(out)
			torch.cuda.empty_cache()
		
		batch_size, channels, dims, lengths = out.size()
		out = out.permute(3, 0, 1, 2)
		
		out = torch.reshape(out, [lengths, batch_size, channels*dims])
		out, _ = self.enhance(out)
		out = self.tranform(out)
		out = torch.reshape(out, [lengths, batch_size, channels, dims])

		out = out.permute(1, 2, 3, 0)
		
		for idx in range(len(self.decoder)):
			wanted_inx = len(self.encoder) - 1 - idx
			for i in range(wanted_inx+1):
				layer = self.encoder[i]
				out0 = layer(out0)
			wanted_out = out0

			#out = complex_cat( [out, encoder_out[-1 - idx]], 1 )
			out = complex_cat( [out, wanted_out], 1 )

			del wanted_out, out0

			out = self.decoder[idx](out)
			out = out[... , 1:]
		
		#del encoder_out

		mask_real = out[:, 0]
		mask_imag = out[:, 1] 

		del out

		mask_real = F.pad(mask_real, [0, 0, 1, 0])
		mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
		
		mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
		real_phase = mask_real / (mask_mags + 1e-8)
		imag_phase = mask_imag / (mask_mags + 1e-8)

		del mask_real, mask_imag

		mask_phase = torch.atan2(imag_phase, real_phase) 

		del imag_phase, real_phase

		mask_mags = torch.tanh(mask_mags)
		est_mags = mask_mags * spec_mags
		est_phase = spec_phase + mask_phase

		del mask_mags, spec_phase, mask_phase

		real = est_mags * torch.cos(est_phase)
		imag = est_mags * torch.sin(est_phase)

		del est_mags, est_phase
		
		out_spec = torch.cat([real, imag], 1) 

		del real, imag

		out_wav = self.istft(out_spec)

		out_wav = torch.squeeze(out_wav, 1)
		#out_wav = torch.tanh(out_wav)
		out_wav = torch.clamp_(out_wav, -1, 1)
		return out_spec, out_wav

	def get_params(self, weight_decay=0.0):
		# add L2 penalty
		weights, biases = [], []
		for name, param in self.named_parameters():
			if 'bias' in name:
				biases += [param]
			else:
				weights += [param]
		params = [{
					'params': weights,
					'weight_decay': weight_decay,
				}, {
					'params': biases,
					'weight_decay': 0.0,
				}]
		return params

	def loss(self, inputs, labels, loss_mode='SI-SNR'):
		if loss_mode == 'MSE':
			b, d, t = inputs.shape 
			labels[:, 0, :] = 0
			labels[:, d//2, :] = 0
			return F.mse_loss(inputs, labels, reduction='mean') * d

		elif loss_mode == 'SI-SNR':
			return -(si_snr(inputs, labels))
		
		elif loss_mode == 'MAE':
			gth_spec, gth_phase = self.stft(labels) 
			b, d, t = inputs.shape 
			return torch.mean(torch.abs(inputs-gth_spec)) * d
