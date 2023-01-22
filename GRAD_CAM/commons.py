import io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import timm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
	class CustomEfficientNet(nn.Module):
		def __init__(self, model_name = 'tf_efficientnet_b0_ns', pretrained=False):
				super().__init__()
				self.model = timm.create_model(model_name, pretrained = pretrained)
				n_features = self.model.classifier.in_features
				self.model.classifier = nn.Linear(n_features, 2)

		def forward(self, x):
				x = self.model(x)
				return x
		
	model = CustomEfficientNet('tf_efficientnet_b0_ns', pretrained=False)
	weights_path = 'tf_efficientnet_b0_ns_fold0_best.pth'
	model.load_state_dict(torch.load(weights_path, map_location = device)['model'], strict=True)
	return model

def get_tensor(image_bytes):
	my_transforms = transforms.Compose([
                            transforms.Resize(256),
        				    transforms.ToTensor(),
        				    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             					  std=[0.229, 0.224, 0.225])])
	image = Image.open(io.BytesIO(image_bytes))
	image.save("static/model_photos/original.jpg")
	return my_transforms(image).unsqueeze(0)