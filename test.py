import torch
import soundfile as sf

from model.dccrn_e import DCCRN

input_audio_path = "./test.wav"
output_audio_path = "./test_out.wav"

weight_path = "./model_weight/2023-09-10_00-44-42_epoch15_weight.pt"
device = "cpu"

def main():
	# torch设置
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
	
	model = DCCRN()
	model.load_state_dict(torch.load(weight_path))
	model.to(device)
	model.eval()

	input_audio, sr_input = sf.read(input_audio_path)

	input_audio = torch.from_numpy(input_audio).to(device).float()

	with torch.no_grad():
		spec, output_audio = model(input_audio)

	sf.write(output_audio_path, output_audio, sr_input)

if __name__ == '__main__':
	main()