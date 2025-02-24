import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Define input tensor (FP32)
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])



# Define QuantizeLinear node
abs_node = helper.make_node(
    "Abs",
    inputs=["input"],
    outputs=["abs_output"]
)

# Define an output tensor
output_tensor = helper.make_tensor_value_info("abs_output", onnx.TensorProto.FLOAT, [1, 10])

# Create the graph
graph = helper.make_graph(
    nodes=[abs_node],
    name="AbsModel",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# Create the model
model = helper.make_model(graph)

# Save the model
onnx.save(model, "abs.onnx")

print("ONNX model with 'abs' saved as 'abs.onnx'")
