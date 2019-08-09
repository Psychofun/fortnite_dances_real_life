import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image


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
		
		img = cv2.imread(str(img_path))
		
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
		
		mask  = label[:,:] > 0 

		intensity = .80
		img[mask,:] = int(255*intensity)
		

		fig.clear()
		plt.axis('off')
	
		plt.imshow(img)
		



		


def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency 
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL 
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im

if __name__ == '__main__':
		
	

	#NETWORK CREATION 
	weight_name = openpose_dir.joinpath('network/weight/pose_model.pth')

	model = get_model('vgg19')     
	model.load_state_dict(torch.load(weight_name))
	model = torch.nn.DataParallel(model).cuda()
	model.float()
	model.eval()

	img_dir = Path('../../data/source/images')
	NUM_FRAMES = len(os.listdir(str(img_dir)))
	FPS = 30

	plt.close()
	plt.axis('off')
	fig = plt.figure(figsize=(5.12, 5.12))
	
	make_video_animation(num_frames = NUM_FRAMES,
						animation_function = skeleton_frame,
						output_name = "test_labels",
						fps = FPS)
    
	