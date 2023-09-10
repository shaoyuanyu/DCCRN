import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tensorboardX import SummaryWriter

from model.dccrn_e import DCCRN
from Datasets import PrimewordsMD2018
from PMU.gpu_mem_track import MemTracker

load_weight = True
weight_path = "./model_weight/2023-09-09_11-02-42_epoch5_weight.pt"

dataset_path = "../dataset/primewords_md_2018_set1"
noises_path = "../dataset/noises/dormitory-40"

device = "cuda:0"
batch_size = 48
epoch = 1000
learning_rate = 1e-2

# 验证
def validateModel(model, validate_loader):
	model.eval()

	correct = 0
	loss = 0
	total = 0

	with torch.no_grad():
		for x, label in validate_loader:
			x = x.float().to(device)
			label = label.float().to(device)

			spec, pred = model(x)
			del spec

			total += 1
			loss += model.loss(pred, label, loss_mode='SI-SNR').item()

			del pred

	model.train()

	correct = 100 - loss/total

	return correct

def main():
	# torch设置
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True

	# gpu tracker
	gpu_tracker = MemTracker()

	#
	train_dataset = PrimewordsMD2018(dataset_path="{}/train_data".format(dataset_path), noise_path=noises_path, use_rate=0.1)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	validate_dataset = PrimewordsMD2018(dataset_path="{}/test_data".format(dataset_path), noise_path=noises_path, use_rate=0.1)
	validate_loader = DataLoader(validate_dataset, batch_size=batch_size)

	# DCCRN-E 论文中宣称的最适合用于real-time的
	model = DCCRN()

	# 加载权重
	if load_weight:
		print("加载权重文件... (路径: {})".format(weight_path))
		model.load_state_dict(torch.load(weight_path))

	model.to(device)

	model.train()

	# 损失函数
	# model内置 SI-SNR

	# 优化器 SGD
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	# 获取训练开始时间
	# 在服务器上运行时，遇到了环境时区不同导致时间不对的问题，先简单粗暴地修改一下
	os.environ['TZ'] = 'Asia/Shanghai'
	time.tzset()
	train_start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
	os.mkdir("./saved_weight/{}/".format(train_start_time))

	tensorboard_writer = SummaryWriter('running_data/{}'.format(train_start_time))
	#tensorboard_writer = SummaryWriter()
	tensorboard_writer.add_text("start test", "started")

	total_step = 0

	# 训练
	for epoch_i in range(epoch):
		print("epoch {}/{} 训练中...".format(epoch_i+1, epoch))
		tensorboard_writer.add_scalar("epoch", epoch_i+1, global_step=total_step)
		tensorboard_writer.add_text("current epoch", str(epoch_i+1))

		for step, (x, label) in enumerate(train_loader):
			# 显示进度
			#if step % 2 == 0:
			#    print("#", end="")
			if step == 0:
				print("  数据读取完成,开始训练...")
			
			total_step += 1
			
			# 数据指定device
			x = x.to(device)
			label = label.to(device)

			# 数据处理(模型weight数据为torch.cuda.DoubleTensor)
			x = x.float()
			label = label.float()

			# 预测
			spec, pred = model(x)
			del spec

			# 计算loss
			loss = model.loss(pred, label, loss_mode='SI-SNR')

			# 梯度置零
			optimizer.zero_grad()

			# 反向传播
			loss.backward()

			# 参数更新
			optimizer.step()

			# 清理显存
			torch.cuda.empty_cache()

			# tensorboard数据可视化
			tensorboard_writer.add_scalar("loss", loss.item(), global_step=total_step)
			tensorboard_writer.add_scalar("acc", (100-loss).item(), global_step=total_step)

			if step % 20 == 0:
				print("--step {} - loss: {}".format(step+1, loss))
		
		print()

		del x, label, pred
		
		if (epoch_i + 1) % 5 == 0:
			# 保存权重
			try:
				torch.save(model.state_dict(), "./saved_weight/{0}/{0}_epoch{1}_weight.pt".format(train_start_time, epoch_i+1))
			except Exception as e:
				print(e)
				torch.save(model.state_dict(), "./saved_weight/{0}_epoch{1}_weight.pt".format(train_start_time, epoch_i+1))

			# 验证
			print("validating...")
			acc_train = validateModel(model, train_loader)
			acc_val = validateModel(model, validate_loader)
			print("第 {} 轮训练\n  训练集准确率: {}\n  验证集准确率: {}".format(epoch_i+1, acc_train, acc_val))

			# tensorboard数据可视化
			try:
				tensorboard_writer.add_scalar("acc_train", acc_train, global_step=epoch_i+1)
				tensorboard_writer.add_scalar("acc_val", acc_val, global_step=epoch_i+1)
			except:
				try:
					tensorboard_writer.add_scalar("acc_train", float(acc_train), global_step=epoch_i+1)
					tensorboard_writer.add_scalar("acc_val", float(acc_val), global_step=epoch_i+1)
				except Exception as e:
					print(e)
					with open('error_log.txt', 'a', encoding='utf-8') as f:
						f.write("{}: \n{}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), e))
						f.close()
		
		# 更新数据集
		if (epoch_i+1) % 10 == 0:
			train_dataset = PrimewordsMD2018(dataset_path="{}/train_data".format(dataset_path), noise_path=noises_path, use_rate=0.1)
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			validate_dataset = PrimewordsMD2018(dataset_path="{}/test_data".format(dataset_path), noise_path=noises_path, use_rate=0.1)
			validate_loader = DataLoader(validate_dataset, batch_size=batch_size)
	
	tensorboard_writer.export_scalars_to_json("./running_data/{}".format(train_start_time))
	tensorboard_writer.close()

if __name__ == '__main__':
	try:
		main()
	except Exception as e:
		print(e)
		with open('error_log.txt', 'a', encoding='utf-8') as f:
			f.write("{}: \n{}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), e))
			f.close()