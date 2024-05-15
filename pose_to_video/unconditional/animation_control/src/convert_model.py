import os
import json

import numpy as np
from tensorflow.keras.models import load_model

from model import get_model, INPUT_DIMENSION

model = load_model("model.ckpt")  # load stateless model
model.build(input_shape=(1, 1, INPUT_DIMENSION))

new_model = get_model(stateful=True)  # create stateful model
new_model.build(input_shape=(1, 1, INPUT_DIMENSION))

for nb, layer in enumerate(model.layers):
    new_model.layers[nb].set_weights(layer.get_weights())

new_model.predict(np.random.randn(1, 1, INPUT_DIMENSION))  # Set input shapes

new_model.save("model.h5")

os.system("rm -r web_model")
os.system("tensorflowjs_converter --input_format=keras model.h5 web_model")

with open("web_model/model.json", "r", encoding="utf-8") as file:
    model_info = json.load(file)

model_info["modelTopology"]["model_config"]["config"]["layers"][0]["config"]["batch_input_shape"] = [1, 1, INPUT_DIMENSION]
for layer in model_info["modelTopology"]["model_config"]["config"]["layers"]:
    if "stateful" in layer["config"]:
        layer["config"]["stateful"] = True

with open("web_model/model.json", "w", encoding="utf-8") as file:
    json.dump(model_info, file)
