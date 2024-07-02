from segmentation_models_pytorch import Unet
from torch.utils import model_zoo
import re

def create_model():
    models = {"url":"unet_resnet34.pth", "model":Unet(encoder_name="resnet34", classes=1, encoder_weights=None)}

    model = models["model"]
    state_dict = model_zoo.load_url(models["url"], progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model


def rename_layers(state_dict, rename_in_layers):
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)
        result[key] = value

    return result