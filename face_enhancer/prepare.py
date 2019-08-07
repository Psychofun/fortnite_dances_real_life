from pathlib import Path 
import os 
import cv2 
from tqdm import tqdm

def create_dir(path):
    """
    path: string 
        path to create
    return path object 
    """
    
    p = Path(path)

    # Create 
    try:
        p.mkdir(exist_ok = True)
    
    except FileExistsError:
        print("Dir exists. Abort creation. . .")
    

    return p 


face_sync_dir = create_dir('../data/face')

test_sync_dir = create_dir('../data/face/test_sync')

test_real_dir = create_dir('../data/face/test_real')

test_img = create_dir('../data/target/test_img')

test_label = create_dir('../data/target/test_label')


train_dir  = '../data/target/train/train_img'
label_dir =  '../data/target/train/train_label'

print('Prepare test_real....')

train_imgs = os.listdir(train_dir)

for img_idx in tqdm(range(len( train_imgs))):

    path_img = os.path.join( train_dir,'{:05}.png'.format(img_idx))
    img = cv2.imread(  path_img  )
    
    path_label = os.path.join(  label_dir , '{:05}.png'.format(img_idx) )
    label = cv2.imread(path_label)
    
    path_test_real = os.path.join( str(test_real_dir), '{:05}.png'.format(img_idx))
    cv2.imwrite(   path_test_real, img )

    print("Path test real", path_test_real)

    path_test = os.path.join( str(test_img) ,  '{:05}.png'.format(img_idx) )
    cv2.imwrite(path_test , img)

    path_test_label = os.path.join(str(test_label), '{:05}.png'.format(img_idx))
    cv2.imwrite(path_test_label, label)


print('Prepare test_sync. . .')
import torch 
from collections import OrderedDict
import sys 

sys.path.append( str(Path('../src/'  )))
pix2pixhd_dir = Path('../src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))



from pix2pixHD.data.data_loader import CreateDataLoader
from pix2pixHD.models.models import create_model
import pix2pixHD.util.util as util
from pix2pixHD.util.visualizer import Visualizer
from pix2pixHD.util import html 

from  config import test_opt as opt


#print("CWD",os.getcwd())
#print("Path",*sys.path, sep = "\n")




os.environ['CUDA_VISIBLE_DEVICES'] = "0"
opt.checkpoints_dir = '../checkpoints/'
opt.dataroot='../data/target/'
opt.name='target'
opt.nThreads=0
opt.results_dir='./prepare/'






iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)

for data in tqdm(dataset):
    minibatch = 1
    generated = model.inference(data['label'], data['inst'])

    visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()

print('Copy the synthesized images...')
synthesized_image_dir = './prepare/target/test_latest/images/'
for img_idx in tqdm(range(len(os.listdir(synthesized_image_dir)))):
    img = cv2.imread(synthesized_image_dir+' {:05}_synthesized_image.jpg'.format(img_idx))
    cv2.imwrite(str(test_sync_dir) + '{:05}.png'.format(img_idx), img)