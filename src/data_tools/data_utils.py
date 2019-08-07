import torch
from src import utils

class DataPrefetcher():

    def __init__(self, loader):
        self.pool = torch.nn.AvgPool2d(2, 2)
        self.original_loader = loader
        self.stream = torch.cuda.Stream()
        self.i = 0

    def preload(self):
        try:
            self.next_image = next(self.loader)
        except StopIteration:
            self.next_image = None
            return
        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True).float()
            
            self.next_image = interpolate_image(self.pool,
                                                self.next_image,
                                                self.transition_variable)
            
            self.next_image = self.next_image / 255
            self.next_image = self.next_image*2 - 1

    def __len__(self):
        return len(self.original_loader)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_image = self.next_image
        if next_image is None:
            raise StopIteration
        self.preload()
        return next_image
    
    def __iter__(self):
        self.loader = iter(self.original_loader)
        self.preload()
        return self
    
    def update_next_transition_variable(self, transition_variable):
        self.transition_variable = transition_variable


def interpolate_image(pool, images, transition_variable):
    assert 1 < images.max() <= 255
    y = pool(images) // 1
    y = torch.nn.functional.interpolate(y, scale_factor=2)

    images = utils.get_transition_value(y, images, transition_variable)
    return images


def preprocess_images(image, transition_variable, pool=torch.nn.AvgPool2d(2, 2)):
    image = image * 2 - 1
    image = interpolate_image(pool, image, transition_variable)
    return image


def denormalize_img(image):
    image = (image+1)/2
    image = image.clamp(0, 1)
    return image
