import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto

# Define input tensor shape
input_shape = (2, 3, 4)  # Rank = 3

def effective_index(value, dim_size):
    return dim_size + value if value < 0 else value

# Define starts, ends, axes, and steps with negative values
starts = [0]   # Start from the second last element
ends = [3]     # Stop at the last element
axes = [-1]     # Slice along the last dimension
steps = [1]     # Default step

# Convert negative values to effective indices
effective_axes = [len(input_shape) + a if a < 0 else a for a in axes]
effective_starts = [effective_index(starts[i], input_shape[effective_axes[i]]) for i in range(len(starts))]
effective_ends = [effective_index(ends[i], input_shape[effective_axes[i]]) for i in range(len(ends))]

# Compute output shape
output_shape = list(input_shape)
for i, ax in enumerate(effective_axes):
    output_shape[ax] = max(0, (effective_ends[i] - effective_starts[i] + steps[i] - 1) // steps[i])

# Create ONNX tensors for Slice inputs
starts_tensor = helper.make_tensor("starts", TensorProto.INT64, [len(starts)], effective_starts)
ends_tensor = helper.make_tensor("ends", TensorProto.INT64, [len(ends)], effective_ends)
axes_tensor = helper.make_tensor("axes", TensorProto.INT64, [len(axes)], effective_axes)
steps_tensor = helper.make_tensor("steps", TensorProto.INT64, [len(steps)], steps)

# Define the Slice node
slice_node = helper.make_node(
    "Slice",
    inputs=["input", "starts", "ends", "axes", "steps"],
    outputs=["output"]
)

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
output_tensor= helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output_shape))

# Create the ONNX graph
graph = helper.make_graph(
    [slice_node],
    "SliceGraph",
    [input_tensor],
    [output_tensor],
    initializer=[starts_tensor, ends_tensor, axes_tensor, steps_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name="slice_model")
onnx.save(model, "slice_model.onnx")

print("ONNX Slice model saved as slice_model.onnx")
