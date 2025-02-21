import onnxruntime as ort
import numpy as np
import time

#creating onnx session for "quant_dequant"
session = ort.InferenceSession("quant_dequant.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_data = np.random.rand(1, 10000).astype(np.float32)  # Generate random FP32 input
# input_data = np.array([[0.1, 0.1, 0.2, 0.3, 0.4, 0.2, 0.58, 0.7, 0.9, 0.111]], dtype=np.float32) # Generate random FP32 input



# Measure latency
num_runs = 10  # Number of runs for averaging
start_time = time.time()

for _ in range(num_runs):
    output = session.run(None, {input_name: input_data})

end_time = time.time()

# Compute latency
total_time = (end_time - start_time) * 1000000  # Convert to milliseconds
average_latency = total_time / num_runs

print(f"Total Time: {total_time:.2f} ms")
print(f"Average Latency per Inference: {average_latency:.4f} ns")


# print("input_data:", input_data)
# print("Output:", output)
