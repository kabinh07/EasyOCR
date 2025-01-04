from easyocr.easyocr import Reader
import torch
import json

# path = '/home/polygon/easyocr/EasyOCR/trainer/craft/pretrained_model/CRAFT_clr_amp_29500.pth'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.load(path, map_location = device)
print(torch.cuda.is_available())
# # print(model['craft'].keys())
# new_dict = {}
# for key, value in model['craft'].items():
#     new_dict[key[7:]] = value
# model['craft'] = new_dict
# torch.save(model, path)
reader = Reader(["bn"])
print(reader.readtext('img_135.jpg'))