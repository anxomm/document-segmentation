# Document segmentation and quality improvement
This project aims to build a software to detect a piece of paper from an image
and build a new one with just the paper without any type of mark (pen, coffee...)
and with the perspective adjusted. It has been developed using computer vision
techniques with Python and Scikit-Image.

<img src="https://i.imgur.com/xG3tgf6.jpg" alt="Original image" width="500" height="700"/>
<img src="https://i.imgur.com/fKRfy9G.jpg" alt="Output image" width="500" height="700"/>

## Usage
To use it, you can choose an image from the [Examples](Examples) folder by the
number of the document, or specify a path:

```Shell
python main.py <number_or_path>
```

## System behaviour
The processing is done by solving smaller individual problems, and putting them
together to generate a global solution. The following sections describe each of
the problems identified and how it has been solved.

### Preprocessing
This step is not necessary for images with enough contrast between the piece of
paper and the background. However, for those ones with a light or even white
background it is almost compulsory. 

To enhance the contrasts, a gamma adjustement is applied to the image, so the
edges of the paper become clearer.

### Edges and corner identification
First of all, a downsize is performed. After that, a median filter is applied in
order to remove the text so only the important edges are left. Then, the Canny 
operator and the Hough transformation for lines is applied. We then classify the 
four strongest lines in vertical and horizontal lines. Finally, the intersection 
of the ones in the first group with the ones in the second group is find, those 
are the corners. Extrapolating to the original size of the image, we have the 
coordinates of each of the corners.

<img src="https://i.imgur.com/qaN6RWD.jpg" alt="Edges image" width="250" height="350"/>


### Perspective transformation
To avoid errors due to the aproximation and the distorsion made by the filters, a
little correction is made to the corners so they are a bit inside the paper. 
After this, the projective transformation is made by considering that each of the
corners correspond to each of the corners of the image.

<img src="https://i.imgur.com/OAaJhJy.jpg" alt="Warped image" width="500" height="700"/>


### Paper background detection
To detect whether a pixel is text/mark or the paper itself, a local thresholding
is perform. This get rids of lights, shadows and transparent marks such as 
coffee droplets. 

To differenciate between what is text and what is a mark, the standard deviation
of the three RGB components of each pixel that is not background is done. Notice
that in this step the text that is tinted by, for example coffee, is marked as
color/mark.

<img src="https://i.imgur.com/VUiV3Th.jpg" alt="Thresholding image" width="500" height="700"/>


### Marks removal
As the final step, the background color of each pixel in the color mask is computed
in each of the 8 directions and the center with a median filter. If none of these
medians are color, then the pixel is marked as color, so it is removed. If one of
them is a color, then the saturation of the background and the color is compared,
and if it is the same then the pixel is marked as text, so it is not removed.

After getting the pixels that are marks, an hysteresis is performed to detect the 
surrounding ones that may be color as well, but that due to the perspective, the 
image or the ink itself is has not enough saturation. This process has a big 
disadvantage: text that is "touching" color marks will be removed as well.

<img src="https://i.imgur.com/pioVu1H.jpg" alt="Color mask image" width="500" height="700"/>

