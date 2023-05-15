# ComfyUI-Strimmlarns-Aesthetic-Score
Aesthetic score for ComfyUI

# About
Grade images by using code from [Improved Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) for ComfyUI

# Install
  1. clone repository to custom_nodes
  2. cd into the repository and "pip install -r requirements.txt"
  3. download a model, I have tried: 
      *    https://raw.githubusercontent.com/grexzen/SD-Chad/blob/main/chadscorer.pth
      *    https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/blob/main/ava%2Blogos-l14-linearMSE.pth
  4. put model inside folder "\<ComfyUI dir>/models/aesthetic/". Make dir, its not there
  5. Run ComfyUI and pray

# Nodes

![image](https://raw.githubusercontent.com/strimmlarn/ComfyUI_Strimmlarns_aesthetic_score/main/example/nodes.png)

### Load Aesthetic Model:
You pick a model

### Calculate Aestetic Score:
Calculate image score, needs model and image

### Aesthetic Score Sorter:
Takes 2 images and 2 scores and output 2 images and 2 score where image paired with best score gets the top output

### Score To Number:
Convert the score so it can be used in nodes requiring a number. Alot of stuff in [Was Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui/]) use number as intput

# Example Project
Generate 4 images with diffrent samplers and sort them by aesthetic score. Higest to lowest, json included in image:
  ![image](https://raw.githubusercontent.com/strimmlarn/ComfyUI_Strimmlarns_aesthetic_score/main/example/sort4imagestoptobotton.png)

# Other Projects Similar Topic
  * [Improved Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) 
  * [SD-Chad](https://github.com/grexzen/SD-Chad)
  * [Aesthetic Image Scorer](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer)
  * [Aesthetic Scorer extension for SD Automatic WebUI](https://github.com/vladmandic/sd-extension-aesthetic-scorer)
  * [Simulacra Aesthetic Models ](https://github.com/crowsonkb/simulacra-aesthetic-models)
