import numpy as np
from PIL import Image

def get_images_from_encodings(element, image_processor):
    image_mean = image_processor.image_mean
    image_std = image_processor.image_std

    imgs = []
    for i in range(element["pixel_values"].shape[0]):
        img_tensor = element["pixel_values"][i].squeeze(0)
        unnormalized_image = (img_tensor.cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
        unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        
        imgs.append(Image.fromarray(unnormalized_image))
    return imgs
       
