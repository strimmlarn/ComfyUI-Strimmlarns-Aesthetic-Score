from PIL import Image
import os
from warnings import filterwarnings
import pytorch_lightning as pl
import torch.nn as nn
import torch
from os.path import join
import clip
from PIL import Image#, ImageFile
import folder_paths

# create path to aesthetic model.
folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(folder_paths.models_dir,"aesthetic")], folder_paths.supported_pt_extensions)

#
# Class taken from https://github.com/christophschuhmann/improved-aesthetic-predictor simple_inference.py
#
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)
    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class AestheticModel:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
    return { "required": {"model_name": (folder_paths.get_filename_list("aesthetic"), )}}
  RETURN_TYPES = ("AESTHETIC_MODEL",)
  FUNCTION = "load_model"
  CATEGORY = "aestheticscore"
  def load_model(self, model_name):
    #load model
    m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
    m_path2 = os.path.join(m_path[0],model_name)
    return (m_path2,)

class CalculateAestheticScore:
  def __init__(self):
    pass
  @classmethod
  def INPUT_TYPES(s):
        return {
          "required":{
            "image": ("IMAGE",),
            "aesthetic_model": ("AESTHETIC_MODEL",),
          }
        }
  RETURN_TYPES = ("SCORE",)
  FUNCTION = "execute"
  CATEGORY = "aestheticscore"
  def execute(self, image, aesthetic_model):
    m_path2 = aesthetic_model
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load(m_path2)
    model.load_state_dict(s)
    model.to("cuda")
    model.eval()
    device = "cuda" 
    model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   
    tensor_image = image[0]
    img = (tensor_image * 255).to(torch.uint8).numpy()
    pil_image = Image.fromarray(img, mode='RGB')
    image2 = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
      image_features = model2.encode_image(image2)
      pass
    im_emb_arr = normalized(image_features.cpu().detach().numpy() )
    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
    final_prediction = int(float(prediction[0])*100)
    #hopefully free vram not freezing my computer
    del model
    return (final_prediction,)

class AestheticScoreSorter:
  def __init__(self):
    pass
  pass
  @classmethod
  def INPUT_TYPES(s):
        return {
          "required":{
            "image": ("IMAGE",),
            "score": ("SCORE",),
            "image2": ("IMAGE",),
            "score2": ("SCORE",),
          }
        }
  RETURN_TYPES = ("IMAGE", "SCORE", "IMAGE", "SCORE",)
  FUNCTION = "execute"
  CATEGORY = "aestheticscore"
  def execute(self,image,score,image2,score2):
    if score >= score2:
      return (image, score, image2, score2,)
    else: 
      return (image2, score2, image, score,)

class ScoreToNumber:
  def __init__(self):
    pass
  pass
  @classmethod
  def INPUT_TYPES(s):
        return {
          "required":{
            "score": ("SCORE",)
            }
          }
  RETURN_TYPES = ("NUMBER", )
  FUNCTION = "convert"
  CATEGORY = "aestheticscore"
  def convert(self,score):
    return (score,)
NODE_CLASS_MAPPINGS = {
    "CalculateAestheticScore": CalculateAestheticScore,
    "LoadAestheticModel":AestheticModel,
    "AestheticScoreSorter": AestheticScoreSorter,
    "ScoreToNumber":ScoreToNumber 
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAestheticModel": "LoadAestheticModel",
    "CalculateAestheticScore": "CalculateAestheticScore",
    "AestheticScoreSorter": "AestheticScoreSorter",
    "ScoreToNumber":"ScoreToNumber"
}
