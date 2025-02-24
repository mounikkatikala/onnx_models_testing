import onnxruntime as ort
import numpy as np
import time

#creating onnx session for "matmul_model"
session = ort.InferenceSession("matmul_model.onnx")

input_nameA = session.get_inputs()[0].name
input_nameB = session.get_inputs()[1].name
output_name = session.get_outputs()[0].name

input_dataA = np.random.randint(0,100, (2, 3), dtype=np.uint32)  # Generate random FP32 input
input_dataB = np.random.randint(0,100, (3, 2), dtype=np.uint32)  # Generate random FP32 input
# input_dataA = np.array([[0.1, 0.1], [0.4, 0.2], [0.7, 0.9]], dtype=np.float32) # Generate random FP32 input
# input_dataB = np.array([[0.1, 0.1], [0.4, 0.2], [0.7, 0.9]], dtype=np.float32) # Generate random FP32 input

# Measure latency
num_runs = 1  # Number of runs for averaging
start_time = time.time()

for _ in range(num_runs):
    output_name = session.run(None, {input_nameA: input_dataA, input_nameB: input_dataB})

end_time = time.time()

# Compute latency
total_time = (end_time - start_time) * 1000  # Convert to milliseconds
average_latency = total_time / num_runs

print(f"Total Time: {total_time:.2f} ms")
print(f"Average Latency per Inference: {average_latency:.4f} ns")


print("input_dataA:", input_dataA)
print("input_dataB:", input_dataB)
print("output_name:", output_name)
