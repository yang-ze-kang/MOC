import torch
import numpy as np
import os
import openslide
import h5py
from models.model_set_mil import MIL_Attention_FC_surv
import pdb
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse

parser = argparse.ArgumentParser(
    description='Configurations for Draw Heatmap.')
parser.add_argument(
    '--id', type=str, default='TCGA-CJ-5672-01Z-00-DX1.E319BB3C-61C0-4324-A448-57B5EC921C17', help='wsi id')
parser.add_argument('--weights', type=str, default=None,
                    help='path to model weights')
parser.add_argument('--wsi_dir', type=str, default=None,
                    help='path to wsi data')
parser.add_argument('--h5_dir', type=str, default=None,
                    help='path to wsi h5 data')


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.2) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    # cam[mask==0] = img[mask==0]
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def create_model(name='amil'):
    if name == 'amil':
        model_dict = {'omic_input_dim': None, 'drop_instance': 0.25,
                      'fusion': None, 'n_classes': 1}
        model = MIL_Attention_FC_surv(**model_dict)
        model.eval()
    else:
        raise NotImplementedError
    return model


def read_wsi(wsi_path, h5_path):
    # path_features = torch.load(pt_path)
    wsi = openslide.OpenSlide(wsi_path)
    assert 'aperio.AppMag' in wsi.properties
    img = wsi.read_region((0, 0), wsi.level_count-1, wsi.level_dimensions[-1])
    with h5py.File(h5_path, 'r') as f:
        coords = np.round(np.array(f['coords'])/wsi.level_downsamples[-1])
        path_features = torch.tensor(f['features'])
    assert max(coords[:, 0]) <= img.size[0] and max(
        coords[:, 1]) <= img.size[1]
    return wsi, np.array(img)[::8, ::8, :3], path_features, np.array(coords, dtype=int)


if __name__ == '__main__':
    args = parser.parse_args()
    id = args.id
    weight_path = args.weights
    wsi_path = os.path.join(args.wsi_dir, f'{id}.svs')
    h5_path = os.path.join(args.h5_dir, f'{id}.h5')
    wsi, raw_img, features, coords = read_wsi(wsi_path, h5_path)
    mag = int(wsi.properties['aperio.AppMag'])
    Image.fromarray(raw_img).save(f'{id}_raw.png')
    print('image size:', raw_img.shape)
    # model
    device = torch.device('cuda')
    model = create_model('amil')
    model.load_state_dict(torch.load(weight_path))
    model = model.to(device)
    features = features.to(device)
    res = model.forward_one_wsi(features)
    risk = res['risk'].cpu().detach().item()
    patch_risk = res['patch_risk'].cpu().detach().numpy()
    attention = res['attention'].cpu().detach().numpy()

    # attention
    img = np.float32(raw_img/256)
    num = len(attention)
    base = int(num/256)
    rem = num - int(base*256)
    x = [[i]*(base+1) for i in range(rem)]
    x.extend([i]*base for i in range(rem, 256))
    x = np.concatenate(x)
    index = np.argsort(attention)
    attention[index] = x/255
    mask_attention = np.zeros(img.shape[:2])
    coords = np.array(coords/8, dtype=int)
    if mag == 40:
        mask_attention[coords[:, 1]*2, coords[:, 0]*2] = attention
        mask_attention[coords[:, 1]*2+1, coords[:, 0]*2+1] = attention
    elif mag == 20:
        mask_attention[coords[:, 1], coords[:, 0]] = attention
    else:
        raise NotImplemented
    img = show_cam_on_image(img, mask_attention, use_rgb=True)
    Image.fromarray(img).save(f'{id}_attention.png')

    # patch_risk
    img = np.float32(raw_img/256)
    mask_risk = np.zeros(img.shape[:2])
    if mag == 40:
        mask_risk[coords[:, 1]*2, coords[:, 0]*2] = patch_risk
        mask_risk[coords[:, 1]*2+1, coords[:, 0]*2+1] = patch_risk
    elif mag == 20:
        mask_risk[coords[:, 1], coords[:, 0]] = patch_risk
        mask_risk[coords[:, 1], coords[:, 0]] = patch_risk
    else:
        raise NotImplemented
    img = show_cam_on_image(img, mask_risk, use_rgb=True)
    Image.fromarray(img).save(f'{id}_risk.png')
    print(f"Predicted Risk:{risk}")
