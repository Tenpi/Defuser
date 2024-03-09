# Defuser

![Image](https://github.com/Tenpi/Defuser/blob/main/assets/images/readme.png?raw=true)

Defuser is an UI for diffusers (an image generation library). The inferencing and training of diffusion 
models is supported. It aims to be simple and fast to use.

### Features
- Generate images from text input.
- Generate images from img input.
- Expand and/or patch images.
- ControlNet processing (canny, depth, lineart, lineart anime, softedge, scribble, reference).
- Segmentate (remove background) from images.
- Extract lineart from images.
- Load checkpoints, vae, textual inversions, hypernetworks, and lora.
- Train textual inversions, hypernetworks, lora, dreambooth, and new checkpoints.
- Merge two or three checkpoints together.
- Interrogate images with WDTagger, DeepBooru, or BLIP.
- Auto tag and source images with interrogator and saucenao.
- Embed all train image sources into model files.
- Save generation parameters as image metadata that can be recalled.
- Upscale generated images using waifu2x, Real-ESRGAN, or Real-CUGAN.
- Watermark tool to apply visual watermarks to generated images.
- Apply invisible watermarks to generated images.
- Save images and prompts and view all images that were generated with a certain prompt.
- Change weight of keywords in the prompt.
- Generate random prompts or pick a random saved prompt.
- Past image history viewer (including on the sidebar).
- Generate a gif with AnimateDiff.
- Detect AI images.
- Train image classifiers.
- Simplify and clean a rough sketch.
- Shade a sketch from different lighting directions.
- Colorize a sketch using reference image.
- Divide an image into PSD layers.
- Cross adapt SD1.5 lora to use in SDXL.
- Change the generation source (local, Novel AI, Holara AI)

### Supported Architectures/Services
- Stable Diffusion 1
- Stable Diffusion 2 ("V2" in checkpoint name)
- Stable Diffusion XL ("XL" in checkpoint name)
- Stable Cascade ("SC" in checkpoint name)

### Requirements

Node.js: https://nodejs.org/en \
Python: https://www.python.org/downloads/ \
Git: https://git-scm.com/downloads \
Git LFS: https://git-lfs.com

### Installation

Clone the repository.
```sh
git clone https://github.com/Tenpi/Defuser.git
cd defuser
```

Download the models from https://huggingface.co/Moepi/models 
and place them in the correct folder in "models". This script 
should download all of them.
```sh
git lfs install
git clone https://huggingface.co/Moepi/models
```

Install the code dependencies.
```sh
npm install
pip3 install -r requirements.txt
```

Install pytorch nightly for latest features. 
https://pytorch.org/get-started/locally/

### Running

Start the server with the following command.
```sh
npm start
```

By default, it will be available at http://localhost:8084.

### Animeface

If you have trouble installing animeface (to crop images to the face) you have to build it from the source.
```sh
git clone https://github.com/nya3jp/python-animeface.git
cd python-animeface/third_party/nvxs-1.0.2
./configure
make
sudo make install
```

### ControlNet

ControlNet lets you control the generation using a "control image". The possible control images are canny, depth,
lineart, anime, manga, softedge, scribble, and original image (reference). By clicking on "reference image", it will
also reference the input image rather than just the control image. "Guess mode" is an option that will try to guess the
content if no prompt is provided.

### X-Adapter

X-Adapter lets you use lora models trained on SD1.5 in SDXL. In the dropdown select a v1 checkpoint that it will cross adapt to.

### Patch Controls

<img height="300" src="https://github.com/Tenpi/Defuser/blob/main/assets/images/patch.png?raw=true"/>

These are the drawing controls when using patch:
- Wheel: brush size
- Q: brush size down
- W: brush size up
- B: brush
- E: eraser

### Prompt Weights

To add weight to a word, append "+". To subtract weight, add "-". You can also append a number for a specific value. If the word
contains spaces, wrap it in parantheses. Examples: \
`red++` \
`blue--` \
`(blue hat)++` \
`(red hat)1.5`

### Lineart Extraction

Invert the controlnet processor to extract the lineart of the image (optionally using alpha).

### Background Removal

Click the transparency icon in the input image to run the segmentator which will attempt to separate the character from the background.

### Watermark Tool

There is a tool for adding a watermark to the images, its recommended to use it to make it clear that the image is AI. There is 
also an invisible watermark that isn't visible but can be decoded from the image.

### Image Sources

You should generate sources for the images in order to embed them into the metadata of the trained model. The rate limit of 
a free saucenao account is 4/30s but you can add them manually by creating a file {image name}.source.txt with the source url. 
Shortcut: Any image with name like 1234567.png will be assumed to be from pixiv with source https://www.pixiv.net/en/artworks/1234567. 

### Miscellaneous Tools
- Simplify Sketch - Attempts to clean up a rough sketch.
- Shade Sketch - Applies shading to a sketch from a configerable light direction.
- Colorize Sketch -Attempts to color a sketch using a reference image.
- Layer Divider - Attempts to divide an image into psd layers.
- Embedding Converter - Converts an SD1.5 embedding to SDXL.
- AI Detector - Detect if an anime-styled image is AI.
- Train Classifier - Train an image classification model.

### Important

By using this UI you agree not to generate harmful or offensive content, plagiarize specific artworks or artists, impersonate 
specific indidividuals, or otherwise any other malicious use. This is intended only for personal/offline usage. Note that models 
other than cc0 mitsua diffusion are prone to generating potential copyright infringing material.

### Related/Credits

- [diffusers](https://github.com/huggingface/diffusers)
