from PIL import Image
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import tqdm
def minmax_size(img: Image, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)] #44,672 ratio: tỉ lệ gữa size ảnh thật và size max
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios) #(44,14)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def pad(img: Image, divable: int = 32) -> Image:
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    threshold = 128
    data = np.array(img.convert('LA'))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(np.uint8)
    else:
        data = (255-data[..., -1]).astype(np.uint8)
    data = (data-data.min())/(data.max()-data.min())*255
    if data.mean() > threshold:
        # To invert the text to white
        gray = 255*(data < threshold).astype(np.uint8)
    else:
        gray = 255*(data > threshold).astype(np.uint8)
        data = 255-data

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    im = Image.fromarray(rect).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded


# img = Image.open("D:\LaTeX-OCR-1\datasets\images_private_test\part4.1_32.jpg")

for path in tqdm.tqdm(glob.glob("D:\Img2Latex\data_train/*.png")):
    img = Image.open(path)
    img = pad(minmax_size(pad(img), (192,672), (32,32)))
    input_image = img.convert('RGB').copy()
    input_image.save(path)

