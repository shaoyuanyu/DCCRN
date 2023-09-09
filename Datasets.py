# YSY
# created at 2023/9/7
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import numpy as np
import os
from random import randint, sample
import math

class PrimewordsMD2018(Dataset):
	"""
	PrimewordsMD2018数据集来自: http://www.openslr.org/47/
	数据集中为干净的普通话音频(clean)
	noise数据集为本人自行录制的噪音音频
	在__getitem__()中通过随机选取noise音频叠加到clean音频上(mixAudio())模拟不同的噪声环境
	"""

	def __init__(self, dataset_path, noise_path, use_rate=1):
		# 数据规范
		self.audio_len = 64000 # 统一音频长度为192000(48k采样率下的4s) #16k下4s

		print("加载PrimewordsMD2018数据集...")

		self.clean_audio_paths = self.readFilePaths(dataset_path)
		self.clean_audio_paths = sample(self.clean_audio_paths, math.ceil(len(self.clean_audio_paths)*use_rate))
		self.noise_audio_paths = self.readFilePaths(noise_path)
		
		print(len(self.clean_audio_paths))

	def __getitem__(self, index):
		clean_audio_path = self.clean_audio_paths[index]
		noise_audio_path = self.noise_audio_paths[ randint( 0, len(self.noise_audio_paths)-1 ) ]

		clean_audio, sr_clean = sf.read(clean_audio_path)
		noise_audio, sr_noise = sf.read(noise_audio_path)

		input_audio, label_audio = self.mixAudio(noise=noise_audio, clean=clean_audio, sr_noise=sr_noise, sr_clean=sr_clean)

		del clean_audio, noise_audio, clean_audio_path, noise_audio_path

		input_audio = self.formatAudioLen(input_audio)
		label_audio = self.formatAudioLen(label_audio)

		return input_audio, label_audio
	
	def __len__(self):
		#print("本次加载的数据集长度: {}".format(len(self.clean_audio_paths)))
		return len(self.clean_audio_paths)
	
	# 读取一个路径下的所有文件的路径(可以是多层文件夹下的)
	def readFilePaths(self, dataset_path):
		data_paths = []

		dataset_paths = os.walk(dataset_path)

		for (dir_path, dir_names, file_names) in dataset_paths:
			# 路径格式化
			dir_path = dir_path.replace("\\", "/")

			for file_name in file_names:
				# 文件路径
				data_path = dir_path + "/" + file_name

				# 此处最好加上文件存在性和完整性等项目的检查
				# ...
				
				data_paths.append(data_path)
				#print(data_path)

		return data_paths

	# 统一两音频采样率
	def culSr(self, a, b, sr_a, sr_b):
		"""
		if sr_a != sr_b:
			# 选择较高的采样率作为目标采样率
			sr = max(sr_a, sr_b)

			# 对较低的采样率的音频进行重采样
			if sr_a < sr_b:
				#print("resample noise from {} to {}...".format(sr_a, sr))
				a = librosa.resample(a, orig_sr=sr_a, target_sr=sr)
			else:
				#print("resample clean from {} to {}...".format(sr_b, sr))
				b = librosa.resample(b, orig_sr=sr_b, target_sr=sr)
		else:
			sr = sr_a
		
		return a, b, sr
		"""

		if sr_a != sr_b:
			# 选择较低的采样率作为目标采样率(16k)
			sr = min(sr_a, sr_b)

			# 对较高的采样率的音频进行重采样
			if sr_a > sr_b:
				a = librosa.resample(a, orig_sr=sr_a, target_sr=sr)
			else:
				b = librosa.resample(b, orig_sr=sr_b, target_sr=sr)
		else:
			sr = sr_a
		
		return a, b, sr
	
	# 将双通道音频转为单通道(平均法，将左右声道取均值)
	def singleAudioChannel(self, array2):
		array_l = array2[:, 0]
		array_r = array2[:, 1]

		# 左右声道平均
		array = np.average([array_l, array_r], axis=0)

		return array

	# 将单通道音频转为双通道("1+1"法，将自身通道复制一份)
	def doubleAudioChannel(self, array):
		# 将两个数组堆叠成一个二维数组，表示左右声道
		array2 = np.stack((array, array), axis=1)

		return array2

	def mixAudio(self, noise, clean, sr_noise, sr_clean):
		# 采样率统一
		noise, clean, sr_output = self.culSr(noise, clean, sr_noise, sr_clean)
		
		# 调整noise音频长度
		noise_len = len(noise)
		clean_len = len(clean)

		if noise_len < clean_len:
			# 计算需要重复的次数和余数
			times, remainder = divmod(clean_len, noise_len)
			# 重复noise
			noise = np.concatenate((np.tile(noise, times), noise[0:remainder]))

			#print("叠加noise音源 " + str(times) + " 次，余" + str(remainder))
		
		elif noise_len > clean_len:
			# 裁切noise音源
			noise = noise[0:clean_len]

			#print("裁切noise音源")
		
		# clean格式化为单声道(noise默认为单声道)
		clean_shape = np.shape(clean)

		if len(clean_shape) == 1:
			#单通道
			#clean = self.doubleAudioChannel(clean)
			pass
		elif len(clean_shape) == 2 and clean_shape[1] == 2:
			# 双通道变单声道
			print("clean单声道化")
			clean = self.singleAudioChannel(clean)
		else:
			raise("clean音源通道数出错")

		# 叠加noise和clean音频
		mixed = np.clip(noise+clean, -32768, 32767)

		#print("noise长度: {}, clean长度: {}, mixed长度: {}".format(len(noise), len(clean), len(mixed)))

		return mixed, clean
	
	# 统一音频时长
	def formatAudioLen(self, array):
		array_len = len(array)
		if array_len > self.audio_len:
			start_point = randint(0, array_len-self.audio_len-1)
			array = array[start_point:start_point+self.audio_len]
		elif array_len < self.audio_len:
			#need = self.audio_len - array_len
			#start_point = randint(0, array_len-need-1)
			#array = np.concatenate((array, array[start_point:start_point+need]))
			times, remainder = divmod(self.audio_len, array_len)
			array = np.concatenate((np.tile(array, times), array[0:remainder]))
		
		return array
