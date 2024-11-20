

This is a collection of notebooks and code for segmenting and extracting values from ~ 1200 images of live bee wings
and ~4500 slide mounted bee wings. Not all the analysis is done in these notebooks. Crucially any deep learning
training is done seperately (see folders semantic_seg and age_pred). 

00: Scripts beginning with '00' are either just for exploration (not production) or are used multiple times throughout
  for example, "00. Go Through Images Interactively.ipynb" is used to quickly go through a folder of images and 
  manually flag the images (ie, the segmentation is not very good, or the wing is damaged, etc).

1X: Slide-Mounted Wings: Scripts beginning with this tag are used to primarily analyze the slide-mounted wings, 
  10. Crop Individual Slides : Crop the large TIFs into individual slide images
  11. Wing Segmentation - Segment the wings from the individual slides, flag them as left or right and rotate them
  After this the individual cells are segmented using the UNet++ model trained on the live bees to segment the cells
  17. Make csv file with data - slides: Using the segmentations from the UNet++ model, create a final csv file with the cell sizes, etc.

2X: Live Bees: Scripts beginning with this tag are used to primarily analyze the live bee images.
  20. Find White Card - Find the white card in the image and crop / rotate the image to this - makes it easier to find the wing in the white card
  21. Initial Wing Segmentation - Attempt to segment the wing from the initial image by finding a big dark blob on the card
  22. Coarse Registration using Masks - Attempt to map a wing to the segmentation mask from above using iterative methods
  23. Fine Registration using Segmentations - Use ORB registration to do Affine registration of a typical wing to the segmentation mask
      This allows us to find the wing cell locations in our wings
  23. Map Registered Wing - This allows us to map the registered segmentation back to the original images
  24. Find Blue Line - The blue line in each image is 5mm long, giving us a way to map pixel values to real lengths
  25. Segment Cells using SAM - given the registered cell locations, use SAM to attempt to segment each cell automatically'
  26. Make Segmentation Train Val: Using the best segmentations, create a train/val split to train a UNet on (the UNet is in 'semantic seg' folder)
      and requires different packages, etc. 
  27. Resize UNet Predictions: The UNet model gives the output in a reduced standardized size, so resize the predictions to the original image size
  28. Make csv file with data - live bees: Using the segmentations, create a final csv file with the cell sizes, etc.
  
3X: Age Predictions: Scripts beginning with this tag are used to primarily generate data for or analyze the age prediction results.

## Folders
semantic_seg: This folder contains the script to train the UNet++ model
age_pred: This folder contains the script to train the initial age prediction model 
archive: these are scripts which are no longer relevant or not important for the final product. ie segmenting the slides using SAM, 
      because we actually ended up training a UNet++ model on the live bees and applying those to the slide images.
