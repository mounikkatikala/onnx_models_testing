import onnx
import numpy as np
from onnx import helper, TensorProto

# Define input tensor 
input_tensor = helper.make_tensor_value_info("input", TensorProto.UINT32, [3, 3, 4])

# Create a constant indices tensor 
indices_data = np.array([[1, 2,0]], dtype=np.int64)
indices_tensor = helper.make_tensor(
    name="indices",
    data_type=TensorProto.INT64,
    dims=[indices_data.size],  # Shape of indices
    vals=indices_data.flatten().tolist()
)

# Create the Gather node
gather_node = helper.make_node(
    "Gather",
    inputs=["input", "indices"],
    outputs=["output"],
    axis=1  # Gather along the last dimension (columns)
)

# Extract dimensions
dim0 = input_tensor.type.tensor_type.shape.dim[0]
dim1 = input_tensor.type.tensor_type.shape.dim[1]
dim2 = input_tensor.type.tensor_type.shape.dim[2]

# Extract the axis value from the node
axis_value = next(attr for attr in gather_node.attribute if attr.name == "axis").i

# Extract dimensions
indices_dims = indices_data.size

# Convert to integer values
dim0 = indices_dims if axis_value == 0 else dim0.dim_value 
dim1 =  indices_dims if axis_value == 1 else dim1.dim_value  
dim2 =  indices_dims if axis_value == 2 else dim2.dim_value   


# Define output tensor 
output_tensor = helper.make_tensor_value_info("output", TensorProto.UINT32, [dim0, dim1, dim2])


# Define the graph
graph_def = helper.make_graph(
    nodes=[gather_node],
    name="GatherModel",
    inputs=[input_tensor],  # Only the input tensor is dynamic
    outputs=[output_tensor],
    initializer=[indices_tensor]  # Set indices as a constant
)

# Define the model
model_def = helper.make_model(graph_def, producer_name="GatherModelExample")

# Save the model
onnx.save(model_def, "gather_axis.onnx")

print("Fixed ONNX model 'gather_axis.onnx' created successfully!")
