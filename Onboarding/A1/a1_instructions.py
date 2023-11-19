'''
HW: Satellite Image Classification
You may work on this assignment with ONE partner.

We use the SAT-4 Airborne Dataset: https://www.kaggle.com/datasets/crawford/deepsat-sat4.
Download "sat-4-full.mat" from Kaggle and place it in your working dir.

This dataset is large (500K satellite imgs of the Earth's surface).
Each img is 28x28 with 4 channels: red, green, blue, and NIR (near-infrared).
The imgs are labeled to the following 4 classes: 
barren land | trees | grassland | none

The MAT file from Kaggle contains 5 variables:
- annotations (explore this if you want to)
- train_x (400K training images), dim: (28, 28, 4, 400000)
- train_y (400k training labels), dim: (4, 400000)
- test_x (100K test images), dim: (28, 28, 4, 100000)
- test_y (100K test labels), dim: (4, 100000)

For inputs (train_x and test_x):
0th and 1st dim encode the row and column of pixel.
2nd dim describes the channel (RGB and NIR where R = 0, G = 1, B = 2, NIR = 3).
3rd dim encodes the index of the image.

Labels (train_y and test_y) are "one-hot encoded" (look this up).

Your task is to develop two classifiers, SVMs and MLPs, as accurate as you can.
'''

# TASK: Load in the dataset
# Note: Use scipy.io.loadmat
# Note: Dealing with 400K and 100K images will take forever.
# Feel free to train and test on small subsets (I did 10K and 2.5K, tune as you need).
# Just make sure your subset is rather uniformly distributed and not biased.
# Once you have your x_train, y_train, x_test, y_test variables (or however you name it),
# I suggest you save these variables using dump, then load them in subsequent runs.
# This will make things much faster as you wouldn't need to load in the full dataset each time.

# TASK: Pre-processing
# You need to figure out how to pass in the images as feature vectors to the models.
# You should not simply pass in the entire image as a flattened vector;
# otherwise, it's very slow and just not really effective
# Instead you should extract relevant features from the images.
# Refer to Section 4.1 of https://arxiv.org/abs/1509.03602, especially first three sentences
# and consider what features you want to extract
# And like the previous task, once you have your pre-processed feature vectors,
# you may want to dump and load because pre-processing will also take a while each time.
# MAKE SURE TO PRE-PROCESS YOUR TEST SET AS WELL!

# TASK: TRAIN YOUR MODEL
# You have your feature vectors now, time to train.
# Again, train two models: SVM and MLP.
# Make them as accurate as possible. Tune your hyperparameters.
# Check for overfitting and other potential flaws as well.

# TASK: Visualizations
# Produce two visualizations, one for SVM and one for MLP.
# These should show the justifications for choosing your hyperparameters to your classifiers,
# such as kernel type, C value, gamma value, etc. for SVM or layer sizes, depths, itersm etc. for MLPs