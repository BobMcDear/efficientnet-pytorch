# EfficientNet in PyTorch


This is an implementation of EfficientNet in PyTorch. You can find the accompanying blog for the code [here](https://python.plainenglish.io/implementing-efficientnet-in-pytorch-part-1-an-overview-1830935e0c8b) </a> and a Colab version of the code [here](https://colab.research.google.com/drive/1zW4yQoNZyg9twfbcs_orwtmQ36u97ETl?usp=sharing).

# Usage

```efficientnet.py/EfficientNet```: This is a generic form of EfficientNet that takes in width and depth scale factors and returns a corresponding scaled version of EfficientNet-B0. These two can be passed in as ```w_factor``` and ```d_factor``` respectively, with default values of 1. For instance, if you set them to 1.1 and 1.2, that would give EfficneNet-B2, while 2 and 3.1 would give EfficientNet-B7. Moreover, ```out_sz``` can be passed to set the output dimension of the final fully-connected layer, with a default of 1000.

For convenience, EfficientNet B0 through B7 can be found under the same file, where each one's name is ```EfficientNetB[number]```.
