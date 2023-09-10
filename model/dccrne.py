import torch
import torch.nn as nn

class DCCRN(nn.Module):
	def __init__(self):
		super(DCCRN, self).__init__()

		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()