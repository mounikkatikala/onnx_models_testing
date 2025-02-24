import onnx
from onnx import helper, TensorProto

# Define input tensors
A = helper.make_tensor_value_info("A", TensorProto.UINT32, [2, 3])
B = helper.make_tensor_value_info("B", TensorProto.UINT32, [3, 2])

# Define output tensor
Y = helper.make_tensor_value_info("Y", TensorProto.UINT32, [2, 2])

# Create MatMul node
matmul_node = helper.make_node(
    "MatMul",
    inputs= ["A", "B"],
    outputs= ["Y"]
)

# Create the graph
graph = helper.make_graph(
    nodes= [matmul_node],  # List of nodes
    name= "MatMul_Model",  # Graph name
    inputs= [A, B],  # Inputs
    outputs= [Y]      # Outputs
)

# Create the ONNX model
model = helper.make_model(graph, producer_name="onnx-matmul")
onnx.save(model, "matmul_model.onnx")

print("ONNX model 'matmul_model.onnx' created successfully!")
