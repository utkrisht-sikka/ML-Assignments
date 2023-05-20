"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch

from misc_functions import get_example_params, convert_to_grayscale, save_gradient_images, preprocess_image
import sys
import numpy as np
from PIL import Image
import pickle
from copy import deepcopy
class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        print(self.model)
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    # Get params
    target_example = 1  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) = get_example_params(target_example)
    # original_image = Image.open(img_path).convert('RGB')
    # Process image
    # prep_img = preprocess_image(original_image)
    
    pretrained_model = pickle.load(open("model.sav","rb"))
    for i in range(10):
        im_path = "cat_images/cat"+str(i)+".jpg"
         
        original_image = deepcopy(np.asarray(Image.open(im_path).convert('RGB')))
        original_image.resize((224,224,3) )
        prep_img = preprocess_image(original_image)
        file_name_to_export = "id"+str(i)
        # Vanilla backprop
        VBP = VanillaBackprop(pretrained_model)
        # Generate gradients
        vanilla_grads = VBP.generate_gradients(prep_img, target_example)
        # Save colored gradients
        save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
        # Convert to grayscale
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        # Save grayscale gradients
        save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
        print('Vanilla backprop completed')
        
    print(type(original_image))
    print(type(prep_img))
    print(type(target_example))
    print(type(file_name_to_export))
    print(type(pretrained_model))
    sys.exit(0)
    
 