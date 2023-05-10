# ComfyUI-Strimmlarns-Aesthetic-Score
Aesthetic score for ComfyUI

# About
Grade images by using code from [Improved Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) for ComfyUI

# Install
  1. pip install -r requirements.txt in same python enviroment you use when using ComfyUI.
  2. download a model, I have tried: 
      *    https://raw.githubusercontent.com/grexzen/SD-Chad/blob/main/chadscorer.pth
      *    https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/blob/main/ava%2Blogos-l14-linearMSE.pth
  3. put model inside folder "\<ComfyUI dir>/models/aesthetic/". Make dir, its not there. 
  4. Run ComfyUI and pray

# Example Project
Generate 4 images with diffrent samplers and sort them by aesthetic score. Higest to lowest:
  ![image](https://raw.githubusercontent.com/strimmlarn/ComfyUI_Strimmlarns_aesthetic_score/main/example/sort4imagestoptobotton.png)

# Other Projects Similar Topic
  * [Improved Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) 
  * [SD-Chad](https://github.com/grexzen/SD-Chad)
  * [Aesthetic Image Scorer](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer)
  * [Aesthetic Scorer extension for SD Automatic WebUI](https://github.com/vladmandic/sd-extension-aesthetic-scorer)
  * [Simulacra Aesthetic Models ](https://github.com/crowsonkb/simulacra-aesthetic-models)