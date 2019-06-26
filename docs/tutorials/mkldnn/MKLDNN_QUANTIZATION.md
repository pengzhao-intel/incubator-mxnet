
<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Quantize models for production-level inference with MKL-DNN backend

The Apache MXNet* community delivered quantization approaches to improve performance and reduce the deployment costs for inference. There are two main benefits of lower precision (INT8). First, the computation can be accelerated by lower precision instruction, like VNNI. Second, lower precision data types save memory bandwidth and allow for better cache locality and power savings. The new quantization approach can realize up to a 4x performance speedup in current [AWS* EC2 c5.24xlarge instances](https://amazonaws-china.com/ec2/instance-types/c5/) with [Intel Deep Learning Boost](https://www.intel.ai/intel-deep-learning-boost/#gs.0ngn54) enabled hardware with less than 0.5% accuracy drop.

## Installation and Prerequisites

Installing MXNet with MKL-DNN integration is an easy process. You can follow [How to build and install MXNet with MKL-DNN backend](https://mxnet.incubator.apache.org/tutorials/mkldnn/MKLDNN_README.html) to build and install MXNet with MKL-DNN from source. Also, You can install the release or nightly version via PyPi and pip directly by running:

```
# release version
pip install mxnet-mkl
# nightly version
pip install mxnet-mkl --pre
```

## Image Classification Quantization Demo

A new quantization script [imagenet_gen_qsym_mkldnn.py](https://github.com/apache/incubator-mxnet/blob/master/example/quantization/imagenet_gen_qsym_mkldnn.py) has been designed to launch quantization for image-classification models with Intel® MKL-DNN. This script integrates with [Gluon-CV modelzoo](https://gluon-cv.mxnet.io/model_zoo/classification.html), so that more pre-trained models can be downloaded from Gluon-CV and then converted for quantization. For details, you can refer [Model Quantization with Calibration Examples](https://github.com/apache/incubator-mxnet/blob/master/example/quantization/README.md). Below is a demo usage for ResNet50-V1 quantization.

Use the following command to install [Gluon-CV](https://gluon-cv.mxnet.io/):

```
pip install gluoncv
```

The following command is to download the pre-trained model from Gluon-CV and transfer it into the symbolic model which would be finally quantized. The [validation dataset](http://data.mxnet.io/data/val_256_q90.rec) is available for testing the pre-trained models:

```
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
```

The model would be automatically replaced in fusion and quantization format. It is then saved as the quantized symbol and parameter files in the `./model` directory. The following command is to launch inference.

```
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN

# Launch FP32 Inference 
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --param-file=./model/resnet50_v1-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch INT8 Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=./model/resnet50_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50 --batch-size=64 --num-inference-batches=500 --dataset=./data/val_256_q90.rec --ctx=cpu

# Launch dummy data Inference
python imagenet_inference.py --symbol-file=./model/resnet50_v1-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
python imagenet_inference.py --symbol-file=./model/resnet50_v1-quantized-5batches-naive-symbol.json --batch-size=64 --num-inference-batches=500 --ctx=cpu --benchmark=True
```

## Integrate Quantization Flow to Your Project

It's important to note that the quantization flow only work with the symbolic MXNet API. If you're using Gluon, you can first refer [Saving and Loading Gluon Models](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html) to hybridize your computation graph and export it as a symbol before running quantization.

Take Gluon ResNet50 as an example.

### Model Initialization

```python
import logging
import mxnet as mx
from mxnet.gluon.model_zoo import vision
from mxnet.contrib.quantization import *

logging.basicConfig()
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

batch_shape = (1, 3, 224, 224)
resnet50 = vision.resnet50_v1(pretrained=True)
resnet50.hybridize()
resnet50.forward(mx.nd.zeros(batch_shape))
resnet50.export('resnet50_v1')
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet50_v1', 0)
```
First, we download ResNet50-v1 model from gluon modelzoo and export it as a symbol.

### Model Fusion

```python
sym = sym.get_backend_symbol('MKLDNN_QUANTIZE')
```
It's important to add this line to enable graph fusion before quantization to get better performance.

### Float32 Inference

```python
# download imagenet validation dataset
mx.test_utils.download('http://data.mxnet.io/data/val_256_q90.rec', 'quantization/dataset.rec')
# set rgb info for data
mean_std = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939, 'std_r': 58.393, 'std_g': 57.12, 'std_b': 57.375}
# set batch size
batch_size = 16
# create DataIter
data = mx.io.ImageRecordIter(path_imgrec='quantization/dataset.rec', batch_size=batch_size, data_shape=batch_shape[1:], rand_crop=False, rand_mirror=False, **mean_std)
# create module
mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
mod.bind(for_training=False, data_shapes=data.provide_data, label_shapes=None)
mod.set_params(arg_params, aux_params)
# forward inference
for batch in data:
    mod.forward(data_batch=batch, is_train=False)
```
You need to prepare a set of code for float32 inference in symbolic mode.

### Calibration

```python
mx.test_utils.download('http://data.mxnet.io/data/val_256_q90.rec', 'dataset.rec')
mean_std = {'mean_r': 123.68, 'mean_g': 116.779, 'mean_b': 103.939, 'std_r': 58.393, 'std_g': 57.12, 'std_b': 57.375}
batch_size = 16
data = mx.io.ImageRecordIter(path_imgrec='dataset.rec', batch_size=batch_size, data_shape=batch_shape[1:], rand_crop=False, rand_mirror=False, **mean_std)
mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.cpu())
mod.bind(for_training=False, data_shapes=data.provide_data, label_shapes=None)
mod.set_params(arg_params, aux_params)

# exclude layers which do not need quantize
excluded_names = []
# set calib mode.
calib_mode = 'naive'
# set calib_layer
calib_layer = None
# set quantized_dtype
quantized_dtype = 'auto'
# set num_calib_batches
num_calib_batches = 5
max_num_examples = num_calib_batches * batch_size
logger.info('Quantizing FP32 model ResNet50-V1')
qsym, qarg_params, aux_params, collector = quantize_graph(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                          excluded_sym_names=excluded_names,
                                                          calib_mode=calib_mode, calib_layer=calib_layer,
                                                          quantized_dtype=quantized_dtype, logger=logger)

# set monitor to collect layer statistic information
mod._exec_group.execs[0].set_monitor_callback(collector.collect, monitor_all=True)
# collect layer information based on max_num_examples images
num_batches = 0
num_examples = 0
for batch in data:
    mod.forward(data_batch=batch, is_train=False)
    num_batches += 1
    num_examples += batch_size
    if num_examples >= max_num_examples:
        break
if logger is not None:
    logger.info("Collected statistics from %d batches with batch_size=%d"
                % (num_batches, batch_size))

# write scaling factor into quantized symbol
qsym, qarg_params, aux_params = calib_graph(qsym=qsym, arg_params=arg_params, aux_params=aux_params,
                                            collector=collector, calib_mode=calib_mode,
                                            quantized_dtype=quantized_dtype, logger=logger)
# perform post-quantization fusion
qsym = qsym.get_backend_symbol('MKLDNN_QUANTIZE')
# save quantized model
mx.model.save_checkpoint('quantized-resnet50_v1', 0, qsym, qarg_params, aux_params)
```
Applying quantization by inserting some lines into float32 inference code. Below are some descriptions for params of quantization api.

| param              | type            | description|
|--------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| excluded_sym_names | list of strings | A list of strings representing the names of the symbols that users want to excluding from being quantized.|
| calib_mode         | str             | If calib_mode='none', no calibration will be used and the thresholds for requantization after the corresponding layers will be calculated at runtime by calling min  and max operators. The quantized models generated in this mode are normally 10-20% slower than those with  calibrations during inference.<br>If calib_mode='naive', the min and max values of the layer outputs from a calibration dataset will be directly taken as the thresholds for quantization.<br>If calib_mode='entropy', the thresholds for quantization will be derived such that the KL divergence between the distributions of FP32 layer outputs and  quantized layer outputs is minimized based upon the calibration dataset. |
| calib_layer        | function        | Given a layer's output name in string, return True or False for deciding whether to calibrate this layer.<br>If yes, the statistics of the layer's output will be collected; otherwise, no information of the layer's output will be collected.<br>If not provided, all the layers' outputs that need requantization will be collected.|
| quantized_dtype    | str             | The quantized destination type for input data. Currently support 'int8', 'uint8' and 'auto'.<br>'auto' means automatically select output type according to calibration result.|

### INT8 Inference

Now, you have get a pair of quantized symbol and params file, you can load them to launch inference. If you want to use gluon for int8 inference. You can load them as a SymbolBlock:

```python
quantized_net = mx.gluon.SymbolBlock.imports('quantized-resnet50_v1-symbol.json', 'data', 'quantized-resnet50_v1-0000.params')
quantized_net.hybridize(static_shape=True, static_alloc=True)
batch_size = 1
data = mx.nd.ones((batch_size,3,224,224))
quantized_net(data)
```
