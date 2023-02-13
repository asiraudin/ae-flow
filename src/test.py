
import torch
import numpy as np
from data.datasets import AEFlowDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from models import AEFlow
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm
import argparse
import os

import matplotlib.pyplot as plt

def test(submodel_name, model_path, dataset_path):
	# Your training code goes here
	print("Training submodel:", submodel_name)
	print("Model path:", model_path)
	print("Dataset path:", dataset_path)


	device = "cuda" if torch.cuda.is_available() else "cpu"

	test_dataset = AEFlowDataset(root=dataset_path, train=False,
							transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

	model_name = 'fast_flow'
	#model_name = 'res_net'
	model = AEFlow(model_name, device)
	beta = 0.9

	model.load_state_dict(torch.load(model_path))

	model.eval()
	j = 0
	anom0 = []
	anom1 = []
	recon0 = []
	recon1 = []
	flow0 = []
	flow1 = []
	for i, (x, y) in tqdm(enumerate(test_dataloader)):
		j+=1
		#if i == 50: break
		with torch.no_grad():
			x = x.to(device)
			y = y.to(device)
			x_prim, log_prob, logdet_jac = model(x)
			flow =  (-torch.exp(log_prob.mean()/ (np.log(2) *262144)))
			recon = - ssim(x.detach(), x_prim.detach(), reduction='elementwise_mean')
			
			anomaly_score = beta * flow +  (1 - beta) * recon
			
			
			if y == 1:
				anom1.append(anomaly_score.item())
				recon1.append(recon.item())
				flow1.append(flow.item())
			else:
				anom0.append(anomaly_score.item())
				recon0.append(recon.item())
				flow0.append(flow.item())
				


	_, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
	ax1.hist(anom1, color = 'red', alpha=0.5, label = 'anomalous')
	ax1.hist(anom0, color = 'green', alpha=0.5, label = 'normal')
	ax1.title.set_text('Anomaly Score')

	ax2.hist(recon1, color = 'red', alpha=0.5, label = 'anomalous')
	ax2.hist(recon0, color = 'green', alpha=0.5, label = 'normal')
	ax2.title.set_text('S_recon')

	ax3.hist(flow1, color = 'red', alpha=0.5, label = 'anomalous')
	ax3.hist(flow0, color = 'green', alpha=0.5, label = 'normal')
	ax3.title.set_text('S_flow')

	plt.legend()
	plt.show()
if __name__ == "__main__":
	current_directory = os.getcwd()
	parser = argparse.ArgumentParser(description='Test the ae-flow model (produces distribution histogramms"')
	parser.add_argument('--submodel_name', type=str, required=False, default='fast_flow',  help='fast_flow ou res_net, default fast_flow')
	parser.add_argument('--model_path', type=str, required=False, default=current_directory + '/trained-model.pch', help='Path to the model file')
	parser.add_argument('--dataset_path', type=str, required=False, default=current_directory + '/data/chest_xray', help='filepath to the trained model, default workdir/trained-model.pch')

	args = parser.parse_args()
	submodel_name = args.submodel_name
	model_path = args.model_path
	dataset_path = args.dataset_path

	test(submodel_name, model_path, dataset_path)
	