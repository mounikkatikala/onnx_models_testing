import onnxruntime as ort

#creating onnx session for "quant_dequant"
session = ort.InferenceSession("quant_dequant.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_data = np.random.rand(1, 10).astype(np.float32)  # Generate random FP32 input
output = session.run([output_name], {input_name: input_data})

print("Output:", output)
