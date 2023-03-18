#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks
# 
# ## Project: Write an Algorithm for Landmark Classification
# 
# 
# ### Introduction
# 
# The project folder has the following structure:
# 
# * In the main directory you have this notebook, `cnn_from_scratch.ipynb`, that contains the instruction and some questions you will have to answer. Follow this notebook and complete the required sections in order.
# 
# * In the `src/` directory you have several source files. As instructed in this notebook, you will open and complete those files, then come back to this notebook to execute some tests that will verify what you have done. While these tests don't guarantee that your work is bug-free, they will help you finding the most obvious problems so you will be able to proceed to the next step with confidence.
# 
# * Sometimes you will need to restart the notebook. If you do so, remember to execute also the cells containing the code you have already completed starting from the top, before you move on.
# 
# 
# 
# > <img src="static_images/icons/noun-info-2558213.png" alt="?" style="width:25px"/> Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.
# 
# 
# ### Designing and training a CNN from scratch
# 
# In this notebook, you will create a CNN that classifies landmarks.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 50%.
# 
# Although 50% may seem low at first glance, it seems more reasonable after realizing how difficult of a problem this is. Many times, an image that is taken at a landmark captures a fairly mundane image of an animal or plant, like in the following picture.
# 
# <img src="static_images/train/00.Haleakala_National_Park/084c2aa50d0a9249.jpg" alt="Bird in Haleakalā National Park" style="width: 400px;"/>
# 
# Just by looking at that image alone, would you have been able to guess that it was taken at the Haleakalā National Park in Hawaii?
# 
# An accuracy of 50% is significantly better than random guessing, which would provide an accuracy of just 2% (100% / 50 classes). In Step 2 of this notebook, you will have the opportunity to greatly improve accuracy by using transfer learning to create a CNN.
# 
# Experiment with different architectures, hyperparameters, training strategies, and trust your intuition.  And, of course, have fun!
# 
# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 0: Setting up
# 
# The following cells make sure that your environment is setup correctly, download the data if you don't have it already, and also check that your GPU is available and ready to go. You have to execute them every time you restart your notebook.

# In[2]:


# Install requirements
get_ipython().system('pip install -r requirements.txt | grep -v "already satisfied"')


# In[2]:


from src.helpers import setup_env

# If running locally, this will download dataset (make sure you have at 
# least 2 Gb of space on your hard drive)
setup_env()


# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 1: Data
# 
# In this and the following steps we are going to complete some code, and then execute some tests to make sure the code works as intended. 
# 
# Open the file `src/data.py`. It contains a function called `get_data_loaders`. Read the function and complete all the parts marked by `YOUR CODE HERE`. Once you have finished, test that your implementation is correct by executing the following cell (see below for what to do if a test fails):

# In[3]:


get_ipython().system('pytest -vv src/data.py -k data_loaders')


# You should see something like:
# ```
# src/data.py::test_data_loaders_keys PASSED                               [ 33%]
# src/data.py::test_data_loaders_output_type PASSED                        [ 66%]
# src/data.py::test_data_loaders_output_shape PASSED                       [100%]
# 
# ======================= 3 passed, 1 deselected in 1.81s ========================
# ```
# If all the tests are `PASSED`, you can move to the next section.
# 
# > <img src="static_images/icons/noun-info-2558213.png" alt="?" style="width:25px"/> **What to do if tests fail**
# When a test fails, `pytest` will mark it as `FAILED` as opposed to `PASSED`, and will print a lot of useful output, including a message that should tell you what the problem is. For example, this is the output of a failed test:
# > ```
# >    def test_data_loaders_keys(data_loaders):
# >    
# >       assert set(data_loaders.keys()) == {"train", "valid", "test"}
# E       AssertionError: assert {'tes', 'train', 'valid'} == {'test', 'train', 'valid'}
# E         Extra items in the left set:
# E         'tes'
# E         Full diff:
# E         - {'test', 'train', 'valid'}
# E         + {'tes', 'train', 'valid'}
# E         ?                          +++++++
# >
# > src/data.py:171: AssertionError
# -------------- Captured stdout setup ----------------------------------------------
# Reusing cached mean and std for landmark_images
# Dataset mean: tensor([0.4638, 0.4725, 0.4687]), std: tensor([0.2699, 0.2706, 0.3018])
# =========== short test summary info ===============================================
# FAILED src/data.py::test_data_loaders_keys - AssertionError: The keys of the data_loaders dictionary should be train, valid and test
# > ``` 
# > In the `short test summary info` you can see a short description of the problem. In this case, the dictionary we are returning has the wrong keys. Going above a little, you can see that the test expects `{'test', 'train', 'valid'}` while we are returning `{'tes', 'train', 'valid'}` (there is a missing `t`). So we can go back to our function, fix that problem and test again.
# > 
# > In other cases, you might get an error like:
# > ```
#     def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
#                             weight, bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, bias, self.stride,
# >                       self.padding, self.dilation, self.groups)
# E       RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
# >
# > ../../../../miniconda3/envs/udacity_starter/lib/python3.7/site-packages/torch/nn/modules/conv.py:440: RuntimeError
# > ```
# > Looking at the stack trace you should be able to understand what it is going on. In this case, we forgot to add a `.cuda()` to some tensor. For example, the model is on the GPU, but the data aren't.

# <img src="static_images/icons/noun-question-mark-869751.png" alt="?" style="width:25px"/> **Question:** Describe your chosen procedure for preprocessing the data. 
# - How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
# - Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?

# <img src="static_images/icons/noun-answer-3361020.png" alt=">" style="width:25px"/> **Answer**: My data preprocessing first starts by resizing the image to 256 and then crops it to 224. I picked 224 as the input size because it is the recommended input size for using pytorch's pre-trained models. I decided to augment the dataset via RandAugment, a typical set of augmentations for natural images. I also used a Horizontal and vertical flips to augment the images with the goal to get more data to feed it to the neural network, thus improving test accuracy.

# ### Visualize a Batch of Training Data
# 
# Go back to `src/data.py` and complete the function `visualize_one_batch` in all places with the `YOUR CODE HERE` marker. After you're done, execute the following cell and make sure the test `src/data.py::test_visualize_one_batch` is `PASSED`:

# In[4]:


get_ipython().system('pytest -vv src/data.py -k visualize_one_batch')


# We can now use the code we just completed to get a batch of images from your train data loader and look at them.
# 
# Visualizing the output of your data loader is a great way to ensure that your data loading and preprocessing (including transforms such as rotations, translations, color transforms...) are working as expected.

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
from src.data import visualize_one_batch, get_data_loaders

# use get_data_loaders to get the data_loaders dictionary. Use a batch_size
# of 5, a validation size of 0.01 and num_workers=-1 (all CPUs)
data_loaders = get_data_loaders(5, 0.01, -1)

visualize_one_batch(data_loaders)


# In[11]:


data_loaders['train'].dataset.classes


# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 2: Define model
# 
# Open `src/model.py` and complete the `MyModel` class filling in all the `YOUR CODE HERE` sections. After you're done, execute the following test and make sure it passes:

# In[5]:


get_ipython().system('pytest -vv src/model.py')


# <img src="static_images/icons/noun-question-mark-869751.png" alt="?" style="width:25px"/> **Question**: Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

# <img src="static_images/icons/noun-answer-3361020.png" alt=">" style="width:25px"/> __Answer:__ 
# For the backbone, I decided to go with 4 convolutional layers, Additionally, I have useda batchnorm to speed up the training process, as for the activation function, I decided to go with the ReLU activation function, and maxpool to half the image size. 
# 
# For the head, I have used 3 fully-connected-layers with dropout to avoid overfitting, and the output of the last linear layer will have a 50-dimensional vector corresponding to the 50 classes.

# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 3: define loss and optimizer
# 
# Open `src/optimization.py` and complete the `get_loss` function, then execute the test and make sure it passes:

# In[6]:


get_ipython().system('pytest -vv src/optimization.py -k get_loss')


# Then, in the same file, complete the `get_optimizer` function then execute its tests, and make sure they all pass:

# In[7]:


get_ipython().system('pytest -vv src/optimization.py -k get_optimizer')


# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 4: Train and Validate the Model
# 
# > <img src="static_images/icons/noun-info-2558213.png" alt="?" style="width:25px"/> Testing ML code is notoriously difficult. The tests in this section merely exercise the functions you are completing, so it will help you catching glaring problems but it won't guarantee that your training code is bug-free. If you see that your loss is not decreasing, for example, that's a sign of a bug or of a flawed model design. Use your judgement.
# 
# Open `src/train.py` and complete the `train_one_epoch` function, then run the tests:

# In[8]:


get_ipython().system('pytest -vv src/train.py -k train_one_epoch')


# Now complete the `valid` function, then run the tests:

# In[9]:


get_ipython().system('pytest -vv src/train.py -k valid_one_epoch')


# Now complete the `optimize` function, then run the tests:

# In[10]:


get_ipython().system('pytest -vv src/train.py -k optimize')


# Finally, complete the `test` function then run the tests:

# In[11]:


get_ipython().system('pytest -vv src/train.py -k one_epoch_test')


# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 5: Putting everything together
# 
# Allright, good job getting here! Now it's time to see if all our hard work pays off. In the following cell we will train your model and validate it against the validation set.
# 
# Let's start by defining a few hyperparameters. Feel free to experiment with different values and try to optimize your model:

# In[13]:


batch_size = 64        # size of the minibatch for stochastic gradient descent (or Adam)
valid_size = 0.2       # fraction of the training data to reserve for validation
num_epochs = 25        # number of epochs for training
num_classes = 50       # number of classes. Do not change this
dropout = 0.2          # dropout for our model
learning_rate = 0.001  # Learning rate for SGD (or Adam)
opt = 'adam'            # optimizer. 'sgd' or 'adam'
weight_decay = 0.0     # regularization. Increase this to combat overfitting


# In[14]:


from src.data import get_data_loaders
from src.train import optimize
from src.optimization import get_optimizer, get_loss
from src.model import MyModel

# get the data loaders using batch_size and valid_size defined in the previous
# cell
# HINT: do NOT copy/paste the values. Use the variables instead
data_loaders = get_data_loaders(batch_size, valid_size) # YOUR CODE HERE

# instance model MyModel with num_classes and drouput defined in the previous
# cell
model = MyModel(num_classes, dropout) # YOUR CODE HERE


# Get the optimizer using get_optimizer and the model you just created, the learning rate,
# the optimizer and the weight decay specified in the previous cell
optimizer = get_optimizer(model = model, optimizer = opt, learning_rate=learning_rate, weight_decay=weight_decay) # YOUR CODE HERE

# Get the loss using get_loss
loss = get_loss()# YOUR CODE HERE

optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs=num_epochs,
    save_path="checkpoints/best_val_loss.pt",
    interactive_tracking=True
)


# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 6: testing against the Test Set
# 
# > <img src="static_images/icons/noun-info-2558213.png" alt="?" style="width:25px"/> only run this *after* you have completed hyperpameter optimization. Do not optimize hyperparameters by looking at the results on the test set, or you might overfit on the test set (bad, bad, bad)
# 
# Run the code cell below to try out your model on the test dataset of landmark images. Ensure that your test accuracy is greater than 50%.

# In[15]:


# load the model that got the best validation accuracy
from src.train import one_epoch_test
from src.model import MyModel
import torch

model = MyModel(num_classes=num_classes, dropout=dropout)

# YOUR CODE HERE: load the weights in 'checkpoints/best_val_loss.pt'
model.load_state_dict(torch.load('checkpoints/best_val_loss.pt'))

# Run test
one_epoch_test(data_loaders['test'], model, loss)


# ---
# ## <img src="static_images/icons/noun-advance-2109145.png" alt=">" style="width:50px"/> Step 7: Export using torchscript
# 
# Great job creating your CNN models! Now that you have put in all the hard work of creating accurate classifiers, let's export it so we can use it in our app.
# 
# But first, as usual, we need to complete some code!
# 
# Open `src/predictor.py` and fill up the missing code, then run the tests:

# In[5]:


get_ipython().system('pytest -vv src/predictor.py')


# Allright, now we are ready to export our model using our Predictor class:

# In[1]:


# NOTE: you might need to restart the notebook before running this step
# If you get an error about RuntimeError: Can't redefine method: forward on class
# restart your notebook then execute only this cell
from src.predictor import Predictor
from src.helpers import compute_mean_and_std
from src.model import MyModel
from src.data import get_data_loaders
import torch

data_loaders = get_data_loaders(batch_size=1)

# First let's get the class names from our data loaders
class_names = data_loaders["train"].dataset.classes

# Then let's move the model_transfer to the CPU
# (we don't need GPU for inference)
model = MyModel(num_classes=50, dropout=0.5).cpu()

# Let's make sure we use the right weights by loading the
# best weights we have found during training
# NOTE: remember to use map_location='cpu' so the weights
# are loaded on the CPU (and not the GPU)

# YOUR CODE HERE
model.load_state_dict(torch.load('checkpoints/best_val_loss.pt',map_location='cpu'))

# Let's wrap our model using the predictor class
mean, std = compute_mean_and_std()
predictor = Predictor(model, class_names, mean, std).cpu()

# Export using torch.jit.script
scripted_predictor = torch.jit.script(predictor)# YOUR CODE HERE

scripted_predictor.save("checkpoints/original_exported.pt")


# Now let's make sure the exported model has the same performance as the original one, by reloading it and testing it. The Predictor class takes different inputs than the non-wrapped model, so we have to use a specific test loop:

# In[1]:


import torch

# Load using torch.jit.load
model_reloaded =  torch.jit.load("checkpoints/original_exported.pt") # YOUR CODE HERE


# In[ ]:


from src.predictor import predictor_test

pred, truth = predictor_test(data_loaders['test'], model_reloaded)

