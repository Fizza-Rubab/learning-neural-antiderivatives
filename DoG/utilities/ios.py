import os
import numpy as np
import cv2
import simpleimageio as sio
#import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ldr_extensions = [".jpg", ".png"]
hdr_extensions = [".exr", ".hdr"]
#==============================================


def tensor_to_image(tensor, *, denormalize=True):
    """
    Converts a torch tensor image to a numpy array image.

    Parameters
    ----------
    tensor : torch.Tensor
        The image as a torch tensor of shape (channels, height, width) or (1, channels, height, width).
    denormalize : bool
        If true, transform the data range from [-1, 1] to [0, 1].

    Returns
    -------
    img : np.ndarray
        The image as a numpy array of shape (height, width, channels).
    """

    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("If the image tensor has a batch dimension, it must have length 1.")
        tensor = tensor[0]
    if denormalize:
        tensor = (tensor + 1) * 0.5
    img = tensor.numpy(force=True)
    return np.transpose(img, (1, 2, 0))


# find all images in a directory and return list of full paths
def find_images(path, ext = ldr_extensions + hdr_extensions):
    if not isinstance(ext, list):
        ext = [ext]
    img_files = []
    for file in sorted(os.listdir(path)):
        if any([file.endswith(e) for e in ext]):
            img_files.append(os.path.join(path, file))
    if not img_files:
        print("No images found in", path)
    return img_files

#==============================================

# load an image
def load_image(path, normalize=True, append_alpha=False):

    assert os.path.isfile(path), "Image file does not exist"
    is_hdr = is_hdr_from_file_extension(path)
    flags = (cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) if is_hdr else cv2.IMREAD_UNCHANGED

    img = cv2.imread(path, flags)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize and not is_hdr:
        img = img.astype(np.float32) / 255.
    if append_alpha and img.shape[2] == 3:
        alpha = np.ones_like(img[..., 0:1])
        img = np.concatenate([img, alpha], axis=-1)
    return img

#==============================================

# save an image
def save_image(img, path, channels=3, jpeg_quality=95):
    is_hdr = is_hdr_from_file_extension(path)

    if img.ndim == 2:
        out_img = img[..., None]
    if img.ndim == 3 and img.shape[2] >= 2:
        if channels == 2:
            out_img = np.zeros((*img.shape[0:2], 3))
            out_img[..., 1:3] = img[..., 2::-1]
        if channels == 3:
            out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if channels == 4:
            out_img = cv2.cv2Color(img, cv2.COLOR_RGBA2BGRA)
    if (out_img.dtype == np.float32 or out_img.dtype == np.float64) and not is_hdr:
        out_img = np.clip(out_img, 0, 1) * 255
        out_img = out_img.astype(np.uint8)
    if is_hdr:
        out_img = out_img.astype(np.float32)

    cv2.imwrite(path, out_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


#==============================================

# Check if image format should be one of hdr_extensions
def is_hdr_from_file_extension(file_path):
    extension = os.path.splitext(file_path)[1]
    return extension in hdr_extensions

#==============================================

# get spatial resolution of an image
def get_image_resolution(img):
    return img.shape[:2][::-1]

#==============================================

# sample an image patch at coord (x,y) with size (sx, sy)
def sample_image(img, x, y, sx=1, sy=1):
    assert sx>=1 and sy>=1, "Need a patch of at least one pixel size"
    res = get_image_resolution(img)
    assert 0<=x<=res[0]-sx and 0<=y<=res[1]-sy, f"Sample coordinate {x=},{y=},{sx=},{sy=} outside of image with resolution {res}"
    return img[y:y+sy, x:x+sx, :]

#==============================================

# one function to handle all tev displays
def send_to_tev(name, tensor, gamma = 1.0):
    check_tev_connection()
    # Supporting functions

    # prepare image for display in tev (gamma correction applied)
    def clean_image(img):
        if img.flags['C_CONTIGUOUS']:
            # This condition is necessary to correct how the image is ordered in memory which in turn corrects the "stride" parameter for use inside tev
            img = np.asfortranarray(img)

        if img.ndim == 2: # if grayscale
            img = img[:, :, None]

        if img.ndim > 2 and img.shape[2] == 2: # in case of 2 channels, append a third one
            img = np.dstack((img, np.zeros(img.shape[:-1])))

        img = np.power(img, gamma)
        return np.ascontiguousarray(img)

    #==============================================

    # prepare layered image for display in tev
    def clean_layered_image(layers):
        assert type(layers) is dict, "layers needs to be a dictionary of the form \"layer_name\" : layer"
        layers_clean = {}
        for id, img in layers.items():
            layers_clean[id] = clean_image(img)
        return layers_clean

    #==============================================

    # send an image to tev
    def tev_display_image(name, img):
        with sio.TevIpc() as tev:
            tev.display_image(name, clean_image(img))

    #==============================================

    # send a layered image to tev
    def tev_display_layered_image(name, layers):
        with sio.TevIpc() as tev:
            tev.display_layered_image(name, clean_layered_image(layers))

    #==============================================

    # display a (batched, multi-channel, or both) image tensor (torch / numpy) as a layered image in tev
    def tev_display_tensor(name, tensor):
        # helper function for tev display strings
        def create_str(idx, channels):
            if idx == channels:
                string = f"{idx:02}"
            elif channels - idx == 1:
                string = f"{idx:02},{idx+1:02}"
            elif channels - idx == 2:
                string = f"{idx:02},{idx+1:02},{idx+2:02}"
            else:
                string = f"{idx:02},{idx+1:02},{idx+2:02},{idx+3:02}"

            return string

        # builds the dictionary of RGBA images given a single/multi-channel image/tensor
        def construct_layers(image, batch_name=""):
            layers = {}
            channels = image.shape[2]
            for idx in range(0, channels, 4):
                layers[f"{batch_name}[{create_str(idx, channels-1)}]"] = image[:, :, idx:idx+4]

            return layers

        # process a torch tensor and a ndarray differently before construct_layers
        def process_tensor(tensor):
            return tensor_to_image(tensor.unsqueeze(0)) if torch.is_tensor(tensor) else tensor

        layers = {}
        for idx, layer in enumerate(tensor):
            l = construct_layers(process_tensor(layer), f"B_{idx:02}_")
            layers.update(l)

        tev_display_layered_image(name, layers)

    #==============================================

    if isinstance(tensor, dict):
        try:
            tev_display_layered_image(name, tensor)
        except:
            raise ValueError("Expected a dictionary with key: string; value: ndarray.")

    elif isinstance(tensor, np.ndarray):
        assert tensor.ndim <= 4, "Need numpy array in format (batch (optional), channels, width, height)"
        if tensor.ndim <=3:
            # numpy array has the format (width, height, channels)"
            tev_display_image(name, tensor)
        else:
            # numpy array has the format (batch, width, height, channels)"
            tev_display_tensor(name, tensor)

    elif isinstance(tensor, torch.Tensor):
        assert tensor.dim() == 4, "Need tensor in format (batch, channels, width, height)"
        tev_display_tensor(name, tensor)

    else:
        raise ValueError("Input tensor must be a dictionary, ndarray, or Torch Tensor.")

#==============================================

# Check if tev is initialized
def check_tev_connection():
    import socket
    tev = sio.TevIpc()
    tev._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Attempt to create a socket and connect to the specified host and port
    with tev._socket as s:
        try:
            s.connect((tev._hostname, tev._port))
        except:
            raise ConnectionError("Please Initialize Tev !")

#==============================================

# close an image in tev
def tev_close_image(name):
    with sio.TevIpc() as tev:
        tev.close_image(name)
