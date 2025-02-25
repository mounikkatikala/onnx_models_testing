import onnxruntime as ort
import numpy as np
import time


# Define input tensor shape
input_shape = (2, 3, 4)  # Rank = 3

session = ort.InferenceSession("slice_model.onnx")

input_name = session.get_inputs()[0].name

# input_data = np.random.rand(*input_shape).astype(np.float32)

input_data = np.array([
    # [[10, 11, 12, 13],
    #  [14, 15, 16, 17],
    #  [18, 19, 20, 21]],

    [[22, 23, 24, 25],
     [26, 27, 28, 29],
     [30, 31, 32, 33]],

     [[220, 230, 240, 250],
      [260, 270, 280, 290],
      [300, 310, 320, 330]]
], dtype=np.float32)

outputs = session.run(None, {input_name: input_data})



print("Inference input_shape :", input_shape)
print("Inference input_data:", input_data)
print("Inference output shape:", outputs[0].shape)
print("Inference output:", outputs)