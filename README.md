This is a collection of notebooks and code for segmenting and extracting values from ~ 1200 images of live bee wings
and ~4500 slide mounted bee wings. Not all the analysis is done in these notebooks. Crucially any deep learning
training is done seperately. 

00: Scripts beginning with '00' are either just for exploration (not production) or are used multiple times throughout
  for example, "00. Go Through Images Interactively.ipynb" is used to quickly go through a folder of images and 
  manually flag the images (ie, the segmentation is not very good, or the wing is damaged, etc).

1X: Slide-Mounted Wings: Scripts beginning with this tag are used to primarily analyze the slide-mounted wings, beginning
  with finding the wings in the images using convolution and blob detection, the segmenting the wings and tagging 
  them as left or right, then rotating the wings so they all face more or less the same direction.

2X: Live Bees: Scripts beginning with this tag are used to primarily analyze the live bee images.
  Beginning with segmenting the white cards to easier find the wing in the image, then segmenting the wing, 
  then finding the blue line in the image to relate pixels to real sizes, then registering a reference wing
  to the wings, in an attempt to automatically find the cells in the image, so SAM could be used to
  do initial segmentations. 