import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Define input tensor (FP32)
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])

# Define scale and zero point tensors
scale = helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [0.1])  # Scale factor
zero_point = helper.make_tensor("zero_point", onnx.TensorProto.UINT8, [], [128])  # Zero point

# Define QuantizeLinear node
quant_node = helper.make_node(
    "QuantizeLinear",
    inputs=["input", "scale", "zero_point"],
    outputs=["quantized_output"]
)

# Define DequantizeLinear node
dequant_node = helper.make_node(
    "DequantizeLinear",
    inputs=["quantized_output", "scale", "zero_point"],
    outputs=["dequantized_output"]
)

# Define an output tensor
output_tensor = helper.make_tensor_value_info("dequantized_output", onnx.TensorProto.FLOAT, [1, 10])

# Create the graph
graph = helper.make_graph(
    nodes=[quant_node, dequant_node],
    name="QuantDequantModel",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[scale, zero_point]
)

# Create the model
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# Save the model
onnx.save(model, "quant_dequant.onnx")

print("ONNX model with QuantizeLinear and DequantizeLinear saved as 'quant_dequant.onnx'")
