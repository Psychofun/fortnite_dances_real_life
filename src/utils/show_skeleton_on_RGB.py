import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import torch
from tqdm import tqdm
import os
from pathlib import Path

openpose_dir = Path('../pytorch_Realtime_Multi-Person_Pose_Estimation/')

import sys
sys.path.append(str(openpose_dir))

from save_img import *


# openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose



# Match fps whit fps of  source video.
def make_video_animation(num_frames ,  animation_function, output_name,fps = 30):
	"""
	fps: int 
	   frames per second for output video.
	animation_function:
		function that returs a frame given an index j.
	"""
	metadata = dict(title='Movie Test', artist='IA',
						comment='A.I. Dance Autotune')
	writer = FFMpegWriter(fps=fps, metadata=metadata)

	with writer.saving(fig, output_name +'.mp4', dpi=100):
		for j in tqdm(range(num_frames)):
				animation_function(j)
				writer.grab_frame()
		
	torch.cuda.empty_cache()

def skeleton_frame(idx):
		img_path = img_dir.joinpath('{:05d}.png'.format(idx))
		print("Image path", img_path)
		img = cv2.imread(str(img_path))
		print("Image shape", img.shape)
		shape_dst = np.min(img.shape[:2])
		oh = (img.shape[0] - shape_dst) // 2
		ow = (img.shape[1] - shape_dst) // 2
	
		img = img[oh:oh+shape_dst, ow:ow+shape_dst]
		img = cv2.resize(img, (512, 512))
		multiplier = get_multiplier(img)
		with torch.no_grad():
			paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
		r_heatmap = np.array([remove_noise(ht)
						  for ht in heatmap.transpose(2, 0, 1)[:-1]])\
						 .transpose(1, 2, 0)
		heatmap[:, :, :-1] = r_heatmap
		param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
		label, cord = get_pose(param, heatmap, paf)
		
		
		overlay_image = cv2.addWeighted(img,0.4,label,0.1,0)

		fig.clear()
		fig.axis('off')
		plt.imshow(overlay_image)



if __name__ == '__main__':
		
	#video_path = '../../data/source/file_name.mp4'

	#NETWORK CREATION 
	weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')

	model = get_model('vgg19')     
	model.load_state_dict(torch.load(weight_name))
	model = torch.nn.DataParallel(model).cuda()
	model.float()
	model.eval()

	img_dir = Path('../../data/source/images')
	NUM_FRAMES = 20#len(os.listdir(str(img_dir)))
	FPS = 30

	plt.close()
	plt.axis('off')
	fig = plt.figure(figsize=(6.0, 6.0))
	
	make_video_animation(num_frames = NUM_FRAMES,
						animation_function = skeleton_frame,
						output_name = "test_labels",
						fps = FPS)
    
	