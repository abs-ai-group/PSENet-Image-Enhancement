import glob
import os

import cv2
import torch
import torchvision
from model import UnetTMO


def read_image(path):
    img = cv2.imread(path)[:, :, ::-1]
    img = img / 255.0
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


model = UnetTMO()
state_dict = read_pytorch_lightning_state_dict(torch.load("../pretrained/afifi.pth"))
model.load_state_dict(state_dict)
model.eval()
model.cuda()

ds = "/srv/aic"
dataset_path = os.path.join(ds, "Fisheye_deblur")
for split in ['train', 'val', 'test']:
    input_dir = os.path.join(dataset_path, split, 'images')
    if not os.path.exists(input_dir.replace("Fisheye_deblur", "Fisheye_contrast")):
        os.makedirs(input_dir.replace("Fisheye_deblur", "Fisheye_contrast"))
    images = glob.glob(input_dir + '/*')
    assert len(images) != 0
    for image_name in images:
        camera_number, time, _ = image_name.split("camera")[1].split("_")
        if time in ["N", "E"]:
            print("Process:", image_name)
            image = read_image(image_name).cuda()
            with torch.no_grad():
                output, _ = model(image)
            torchvision.utils.save_image(output, image_name.replace("Fisheye_deblur", "Fisheye_contrast"))
        else:
            continue
