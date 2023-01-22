from commons import get_model, get_tensor
import numpy as np
import torch.nn.functional as F
import cv2
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SaveFeatures():
    """ Extract pretrained activations"""
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv[0,:, :, ].reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img


def get_heatmap(image_bytes):
    model = get_model()

    final_conv = model.model.conv_head
    print(final_conv)
    fc_params = list(model.model._modules.get('classifier').parameters())

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    activated_features = SaveFeatures(final_conv)
    weight = np.squeeze(fc_params[0].cpu().data.numpy())

    tensor = get_tensor(image_bytes)
    output = model(tensor)
    # pred_idx = outputs.max(1)
    pred_idx = output.to('cpu').numpy().argmax(1)
    print(pred_idx)

    heatmap = getCAM(activated_features.features, weight, pred_idx)
    hmap_image = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)
    rgba = cv2.cvtColor(hmap_image, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = 0.4
    plt.imsave('static/model_photos/predicted.jpg', hmap_image, cmap = 'jet')

    return pred_idx

    