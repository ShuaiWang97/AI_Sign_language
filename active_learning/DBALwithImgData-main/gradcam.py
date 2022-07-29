from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import torch
import torchvision
import  torchvision.transforms as transforms
import argparse
import torch.nn as nn
from PIL import Image
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
    
def gradcam(img_path, model,dataset, save_path, num):
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    use_cuda = torch.cuda.is_available()
    ## Use final conv layer for res18
    target_layers = [model.layer4] 

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # Gradcam code only accept 4d rgb image. exetend grey scale to rgb
    input_tensor=input_tensor[:,1,:,:]
    input_tensor = input_tensor[:,None,:,:]
    

    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = GradCAM
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=None,
                            eigen_smooth=None)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    #a = cam_image.tolist()
    #print(a)
    #plt.imshow(cam_image)
    #plt.axis('off')
    #plt.show()
    cv2.imwrite(f'{save_path}/{dataset}_{num}_cam.jpg', cam_image)
    cv2.imwrite(f'{save_path}/{dataset}_{num}_gb.jpg', gb)
    cv2.imwrite(f'{save_path}/{dataset}_{num}_cam_gb.jpg', cam_gb)
    
def run_grad_cam(dataset, model, save_path, iterr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = load_CNN_model(backbone="res18",pretrained_model="ASL_MNIST_imbal",device=device)

    for i in range(10):
        img_path =  f'GradCam/sign_examples/{dataset}/{dataset}_{i}.png'
        gradcam(img_path=img_path, model=model, dataset=dataset, save_path=save_path, num=f'{i}_{iterr}')