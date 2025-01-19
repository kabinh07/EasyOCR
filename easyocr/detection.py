import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import resize_aspect_ratio, normalizeMeanVariance
from .craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    low_text = 0.3
    text_threshold = 0.3
    print(f"printing from detection.py | low_text: {low_text}")
    print(f"printing from detection.py | text_threshold: {text_threshold}")
    print(f"printing from detection.py | link_threshold: {link_threshold}")
    low_text = 0.3
    text_threshold = 0.3
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
        real_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        real_image = Image.fromarray(real_image.astype(np.uint8))
        res_image = Image.fromarray(res_image.astype(np.uint8))
        real_image.save("/home/EasyOCR/example_data/test/data/real_image.jpg")
        res_image.save("/home/EasyOCR/example_data/test/data/resized_image.jpg")

    ratio_h = 1/target_ratio[0]
    ratio_w = 1/target_ratio[1]

    # preprocessing
    x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    out1 = y[:, :, :, 0]
    out2 = y[:, :, :, 1]

    ot1_scaled = cv2.resize((out1[0].cpu().detach().numpy()*255).astype(np.uint8), (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    ot1 = Image.fromarray(ot1_scaled, mode = "L")
    ot2_scaled = cv2.resize((out2[0].cpu().detach().numpy()*255).astype(np.uint8), (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    ot2 = Image.fromarray(ot2_scaled, mode = "L")

    ot1.save("/home/EasyOCR/example_data/test/data/out_reg_image.jpg")
    ot2.save("/home/EasyOCR/example_data/test/data/out_aff_image.jpg")

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    return boxes_list, polys_list

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False):
    net = CRAFT()

    trained_model = '/home/EasyOCR/trainer/craft/exp/custom_data_train/CRAFT_clr_best.pt'

    if device == 'cpu':
        print(f"Loading state dict from: {trained_model}")
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device, weights_only=False)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        print(f"Loading state dict from: {trained_model}")
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device, weights_only=False)['craft']))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None, **kwargs):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars)
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result
