# Simple-GAN-for-Art

Simplest version of GAN that can be used for generating art (_or whatever you want!_)

To install dependencies before running the code, use:
```
pip install -r requirements.txt
```

The dataset that I've used is available at >>> https://www.kaggle.com/ikarus777/best-artworks-of-all-time. 

To make things _automatic_, I've used **opendatasets** library which allows us to download datasets from online sources like Kaggle and Google Drive using a simple Python command.

Remember to put your _kaggle.json_ file and put it into the same directory before proceeding. To know more about Kaggle API >> https://github.com/Kaggle/kaggle-api

To run the code, 
```
python main.py
```

The results will be saved into **generated** directory.

To create a video as output, you can use FFMPEG on the generated directory.
```
ffmpeg -framerate 2 -i generated-images-%04d.png output.mp4
```

## One more important point to note!

I've added the support for Automatic Mixed Precision training which boosts the training time by using Tensor Cores (if you have GPU of compute capability 7.0 or higher, then you'll definitely see performance boost). 

GPUs include RTX GPUs, the V100, and the A100.

If you do not want to enable support for autocasting, change the following:

+ Go to line number 143 in **main.py** and change the line to:

```
loss_d, real_score, fake_score = train.train_discriminator(real_images, opt_d, use_amp = False)
```
+ Go to line number 145 in **main.py** and change the line to:

```
loss_g = train.train_generator(opt_g, use_amp = False)
```

**Disclaimer**: The results might vary from models to models. This is my first try at GAN so I made maybe the simplest version of all, you can try making your own with the help of mine! 
