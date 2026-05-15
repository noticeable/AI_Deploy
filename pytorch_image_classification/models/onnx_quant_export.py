import pathlib


ORT_DYNAMIC_OP_TYPES = ['MatMul', 'Gemm']
POINT_CLOUD_SEGMENTATION_QDQ_SCALE = 1.0 / 128.0
POINT_CLOUD_SEGMENTATION_QDQ_ZERO_POINT = 128


def get_ort_quantization_api():
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except Exception as exc:
        raise RuntimeError(
            'onnxruntime quantization is not available. Please install onnxruntime to export quantized ONNX. '
            f'Original error: {exc}') from exc
    return QuantType, quantize_dynamic


def _get_onnx_api():
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except Exception as exc:
        raise RuntimeError(
            'onnx graph editing is not available. Please install onnx and numpy before patching quantized ONNX. '
            f'Original error: {exc}') from exc
    return np, onnx, TensorProto, helper, numpy_helper


def quantize_onnx_model(float_onnx_path, quantized_onnx_path, backend):
    if backend != 'onnxruntime_dynamic':
        raise ValueError(f'Unsupported export.quantized_onnx_backend: {backend}')
    QuantType, quantize_dynamic = get_ort_quantization_api()
    float_onnx_path = pathlib.Path(float_onnx_path)
    quantized_onnx_path = pathlib.Path(quantized_onnx_path)
    quantize_dynamic(float_onnx_path.as_posix(),
                     quantized_onnx_path.as_posix(),
                     op_types_to_quantize=ORT_DYNAMIC_OP_TYPES,
                     weight_type=QuantType.QInt8)


def patch_segmentation_qdq_input(onnx_path, input_name='points'):
    np, onnx, TensorProto, helper, numpy_helper = _get_onnx_api()
    onnx_path = pathlib.Path(onnx_path)
    model = onnx.load(onnx_path.as_posix())
    graph = model.graph

    if any(node.op_type == 'QuantizeLinear' for node in graph.node) and any(node.op_type == 'DequantizeLinear' for node in graph.node):
        return onnx_path

    qdq_input_name = f'{input_name}_qdq'
    scale_name = f'{input_name}_quant_scale'
    zero_point_name = f'{input_name}_quant_zero_point'
    quantized_name = f'{input_name}_quantized'

    if any(initializer.name == scale_name for initializer in graph.initializer):
        return onnx_path

    scale_initializer = numpy_helper.from_array(np.array([POINT_CLOUD_SEGMENTATION_QDQ_SCALE], dtype=np.float32),
                                                name=scale_name)
    zero_point_initializer = numpy_helper.from_array(np.array([POINT_CLOUD_SEGMENTATION_QDQ_ZERO_POINT], dtype=np.uint8),
                                                     name=zero_point_name)
    graph.initializer.extend([scale_initializer, zero_point_initializer])

    quantized_info = helper.make_tensor_value_info(quantized_name,
                                                   TensorProto.UINT8,
                                                   None)
    dequantized_info = helper.make_tensor_value_info(qdq_input_name,
                                                     TensorProto.FLOAT,
                                                     None)
    graph.value_info.extend([quantized_info, dequantized_info])

    quantize_node = helper.make_node('QuantizeLinear',
                                     inputs=[input_name, scale_name, zero_point_name],
                                     outputs=[quantized_name],
                                     name=f'{input_name}_quantize_linear')
    dequantize_node = helper.make_node('DequantizeLinear',
                                       inputs=[quantized_name, scale_name, zero_point_name],
                                       outputs=[qdq_input_name],
                                       name=f'{input_name}_dequantize_linear')

    rewritten = 0
    for node in graph.node:
        for index, current_input in enumerate(node.input):
            if current_input == input_name:
                node.input[index] = qdq_input_name
                rewritten += 1

    if rewritten == 0:
        raise ValueError(f'Unable to patch ONNX graph: no node consumes input {input_name!r}.')

    graph.node.insert(0, dequantize_node)
    graph.node.insert(0, quantize_node)
    onnx.save(model, onnx_path.as_posix())
    return onnx_path


def make_float_onnx_path(output_path):
    output_path = pathlib.Path(output_path)
    return output_path.with_name(output_path.stem + '.float.onnx')
