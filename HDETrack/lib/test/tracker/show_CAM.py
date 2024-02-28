import cv2
import os
import numpy as np
import torch

def getCAM2(features, img, idx):      
    save_path =  'map_path'                                                                    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # os.path = /home/tcm/PycharmProjects/siamft/pysot_toolkit/trackers
    features = features.to("cpu")
    features = features.squeeze(1).detach().numpy()
    img = cv2.resize(img, (256, 256))
    img = img
    img = np.array(img, dtype=np.uint8)
    # mask = features.sum(dim=0, keepdims=False)
    mask = features
    # mask = mask.detach().cpu().numpy()
    mask = mask.transpose((1, 2, 0))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = cv2.resize(mask, (256,256))
    mask = 255 * mask
    mask = mask.astype(np.uint8)
    heatmap = cv2.applyColorMap(255-mask, cv2.COLORMAP_JET)

    img = cv2.addWeighted(src1=img, alpha=0.6, src2=heatmap, beta=0.4, gamma=0)
    name = '/attn_%d.png' % idx
    cv2.imwrite('map_path' + name, img)


def pltshow(pred_map, name):
    import matplotlib.pyplot as plt
    plt.figure(2)
    pred_frame = plt.gca()
    plt.imshow(pred_map, 'jet')
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    pred_name = os.path.dirname(__file__) + '/response/' + str(name) + '.png'
    plt.savefig(pred_name, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close(100)


if __name__ == '__main__':
    feature = torch.rand(16, 16)
    img = torch.rand(256, 256)
    getCAM2(feature, img)
