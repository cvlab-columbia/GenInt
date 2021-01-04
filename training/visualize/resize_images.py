from PIL import Image
import torch
import glob

def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

def main():

    path_folder = '/home/amogh/columbia/research/NeurIPS/images/imgnet_objnet'
    path_output_folder = '/home/amogh/columbia/research/NeurIPS/images/imgnet_objnet_resized'
    list_path_images = glob.glob(path_folder + '/**/*.*')

    size = 256
    new_width,new_height = (224,224) # Size for center crop

    for p in list_path_images:
        
        img = Image.open(p)
        resized_im = resize(img, size)

        width, height = img.size
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        # Crop the center of the image
        img = img.crop((left, top, right, bottom))

        img_folder_name = os.path.basename(os.path.dirname(p))
        path_save_folder = os.path.join(path_output_folder, img_folder_name)
        path_save_img = os.path.join(path_save_folder,os.path.basename(p))
        img.save(path_save_img)

if __name__ == '__main__':
    main()