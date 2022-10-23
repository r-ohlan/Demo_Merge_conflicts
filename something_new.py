from constants import DIRECTORIES_FOR_ANALYSIS, START_FILE_NUMBER, END_FILE_NUMBER

# Two categories per context and five pictures per category
# This code can be adjusted to reflect your actual data and desired analysis
# Other data can be analyzed by swapping the data directory with your own (make sure you preserve the file structure!)
# Check "constants.py" for variable constants data

all_args = argparse.ArgumentParser(description="Selects the CNN models and analysis we want to run")

# Default arguments for analyses used in Aminoff et al. (2022) 
# Investigate context/category responsiveness in a convolutional neural network by using Pearson's Correlation Coefficient
# HOG and pixel similarity analysis of the image dataset (needs to be set to 1)
all_args.add_argument("-run_net_responses", "--run_net_responses", default=1)
all_args.add_argument("-run_compute_ratios", "--run_compute_ratios", default=1)
all_args.add_argument("-pearson_charts", "--pearson_charts", default=1)
all_args.add_argument("-pearson", '--pearson', default=1)
all_args.add_argument("-confounds", '--confounds', default=0) # different data probably won't have confounds - changed to False
all_args.add_argument("-hog_pixel_similarity", "--hog_pixel_similarity", default=0)

# models
all_args.add_argument("-all_models", "--all_models", default=0)
all_args.add_argument("-alexnet", '--alexnet', default=0)
all_args.add_argument("-alexnet_places365", "--alexnet_places365", default=0)
all_args.add_argument("-vgg16", "--vgg16", default=0)
all_args.add_argument("-vgg19", "--vgg19", default=0)
a
