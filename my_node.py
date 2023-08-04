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

class AesteticModel:
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
    device = "cuda" 
    model2 = None
    preprocess = None
    model = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aesthetic_model": ("AESTHETIC_MODEL",),
            },
            "optional": {
                "keep_in_memory": ("BOOL", {"default": True}),
            }
        }

    RETURN_TYPES = ("SCORE",)
    FUNCTION = "execute"
    CATEGORY = "aestheticscore"

    def execute(self, image, aesthetic_model, keep_in_memory):
        if not self.model2 or not self.preprocess:
            self.model2, self.preprocess = clip.load("ViT-L/14", device=self.device)  #RN50x64 

        m_path2 = aesthetic_model

        if not self.model:
            self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
            s = torch.load(m_path2)
            self.model.load_state_dict(s)
            self.model.to(self.device)
        
        self.model.eval()

        tensor_image = image[0]
        img = (tensor_image * 255).to(torch.uint8).numpy()
        pil_image = Image.fromarray(img, mode='RGB')

        # Use the class variable preprocess
        image2 = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Use the class variable model2
            image_features = self.model2.encode_image(image2)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())
        prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        final_prediction = int(float(prediction[0])*100)

        if not keep_in_memory:
            self.model = None
            self.model2 = None
            self.preprocess = None

        return (final_prediction,)

class AesthetlcScoreSorter:
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
    "LoadAesteticModel":AesteticModel,
    "AesthetlcScoreSorter": AesthetlcScoreSorter,
    "ScoreToNumber":ScoreToNumber 
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAesteticModel": "LoadAesteticModel",
    "CalculateAestheticScore": "CalculateAestheticScore",
    "AesthetlcScoreSorter": "AesthetlcScoreSorter",
    "ScoreToNumber":"ScoreToNumber"
}
