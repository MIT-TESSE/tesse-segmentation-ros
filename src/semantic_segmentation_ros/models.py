###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

import numpy as np
import tensorrt as trt
import time

import pycuda.autoinit
import pycuda.driver as cuda

from semantic_segmentation_ros.utils import pad_image, unpad_image


def get_model(model, weight_file):
    if model == "TRTModel":
        return TesseTRTModel(weight_file)
    else:
        raise ValueError("Currently only TensorRT models are supported")


class TesseTRTModel:
    def __init__(self, onnx_file_path):
        self.trt_model = TRTModel(onnx_file_path)

    def infer(self, image):
        image = pad_image(image, 16, 0)
        image = image.transpose(2, 0, 1).astype(np.float32)
        pred = self.trt_model.infer(image)
        pred = unpad_image(pred, 16, 0)
        return pred


class TRTModel:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    def __init__(self, onnx_file_path):
        """ Model for performing inference with TensorRT.

        Args:
            onnx_file_path (str): Path to ONNX file.
        """
        self.input_shape = None
        self.output_shape = None
        self.engine = self.build_engine(onnx_file_path)
        self.context = self.engine.create_execution_context()

        self.out_cpu, self.in_gpu, self.out_gpu, = self.allocate_buffers(self.engine)

    def build_engine(self, onnx_file_path):
        """ Build TensorRT engine from ONNX file.

        Args:
            onnx_file_path (str): Path to ONNX file.

        Returns:
            tensorrt.ICudaEngine: TensoRT model as described by
                the input ONNX file.
        """
        with trt.Builder(
            self.TRT_LOGGER
        ) as builder, builder.create_network() as network, trt.OnnxParser(
            network, self.TRT_LOGGER
        ) as parser:
            builder.max_workspace_size = 1 << 25
            builder.max_batch_size = 1
            with open(onnx_file_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise ValueError("ONNX file parsing failed")
            self.input_shape = network.get_input(0).shape
            self.output_shape = network.get_output(0).shape
            engine = builder.build_cuda_engine(network)
            return engine

    def infer(self, inputs):
        """ Perform inference.

        Args:
            inputs (np.ndarray): Input array of shape CxHxW

        Returns:
            np.ndarray: Prediction of shape specified by the network.
        """
        assert inputs.shape == self.input_shape, "%s, %s" % (
            inputs.shape,
            self.input_shape,
        )
        inputs = inputs.reshape(-1)
        cuda.memcpy_htod(self.in_gpu, inputs)
        self.context.execute(1, [int(self.in_gpu), int(self.out_gpu)])
        cuda.memcpy_dtoh(self.out_cpu, self.out_gpu)
        return self.out_cpu.reshape(self.output_shape)

    def allocate_buffers(self, engine):
        """ Allocate required memory for model inference

        Args:
            engine (tensorrt.ICudaEngine): TensorRT Engine.

        Returns:
            Tuple[np.ndarray,
                  pycuda._driver.DeviceAllocation,
                  pycuda._driver.DeviceAllocation]
                Host output, device input, device output
        """
        # host cpu memory
        h_in_size = trt.volume(engine.get_binding_shape(0))
        h_out_size = trt.volume(engine.get_binding_shape(1))
        h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
        h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
        in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
        out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)

        # allocate gpu memory
        in_gpu = cuda.mem_alloc(in_cpu.nbytes)
        out_gpu = cuda.mem_alloc(out_cpu.nbytes)
        # stream = cuda.Stream()
        return out_cpu, in_gpu, out_gpu
