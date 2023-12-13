import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from diffusers import RePaintPipeline, RePaintScheduler

class TumorRemoval():
    def __init__(self, seg_thresh=1, random_state=42):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.generator = torch.Generator('cpu').manual_seed(random_state)

        # Creating the segmentation model
        self.seg_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        self.seg_model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
        self.seg_model.to(self.device)
        self.seg_model.load_state_dict(torch.load("FCN_resnet50_seg.pth",  map_location=torch.device('cpu')))
        self.seg_thresh = seg_thresh
    
        # Creating inpainting model
        self.repaint_scheduler = RePaintScheduler.from_pretrained("Hatman/ddpm-celebahq-finetuned-few-shot-universe")
        self.repaint_model = RePaintPipeline.from_pretrained("Hatman/ddpm-celebahq-finetuned-few-shot-universe", 
                                                             scheduler=self.repaint_scheduler).to(self.device)


    def forward(self, image):
        self.seg_model.eval()
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        newdata = self.seg_model(image.to(device))['out']
        mask = (newdata.cpu() < self.seg_thresh).float()
        seg_image = (mask) * image.cpu().detach()
      
        mask_image = transforms.ToPILImage()((mask*255).squeeze().numpy())
        seg_image = transforms.ToPILImage()(seg_image.squeeze())
        repainted_image = self.repaint_model(image=seg_image, 
                                             mask_image=mask_image,
                                             generator=self.generator,
                                             num_inference_steps=30,
                                             eta=1,
                                             jump_length=10,
                                             jump_n_sample=1).images[0]
        output = repainted_image.resize((256,256)).convert('L')
        
        return output