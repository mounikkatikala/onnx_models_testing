import onnxruntime as ort
import numpy as np

# Load the model
session = ort.InferenceSession("gather_axis.onnx")

# Define input tensor (2,3,4)
input_data = np.random.randint(0,100, (3, 3, 4), dtype=np.uint32)  # Generate random FP32 input

# input_data = np.array([
#     [[10, 11, 12, 13],
#      [14, 15, 16, 17],
#      [18, 19, 20, 21]],

#     [[22, 23, 24, 25],
#      [26, 27, 28, 29],
#      [30, 31, 32, 33]],

#      [[220, 230, 240, 250],
#       [260, 270, 280, 290],
#       [300, 310, 320, 330]]
# ], dtype=np.float32)

# Run inference
output = session.run(None, {"input": input_data})[0]

print("Gather input_data:\n", input_data)
print("Gather Output:\n", output)
