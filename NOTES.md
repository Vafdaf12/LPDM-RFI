# Notes
Personal notes about how the LPDM repository works, by Vincent Feistel.

# Using `denoise_config.py`

Suppose you have an WxH image with C color channels. Normally, an image is loaded
by PIL with a shape of WxHxC. PyTorch's `ToTensor` transform converts this into a
CxHxW image, [according to the documentation](https://docs.pytorch.org/vision/master/generated/torchvision.transforms.ToTensor.html).
Furthermore, it also scales the image into a [0.0, 1.0] range.

This scaling is then mapped by the script to a [-1.0, 1.0] range instead, acting
as a standardization of sorts.

The script then performs an [unsqueeze operation](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html),
which transforms the shape into 1xCxHxW. I can only assume this is to trick the
batching mechanism into processing only one image (batch size = 1).


## Adapting to numpy arrays

If instead of a PIL image we would like to use a numpy array read from disk, the
following processing steps are required:

0. **NB:** Pre-processing is required to standardize it before it gets read
1. Read the numpy file from disk into a WxHxC array
2. Swap the W and C axes to get a CxHxW image.
3. Convert this array to a tensor
4. Perform an unsqueeze on the tensor to get 1-sized batch