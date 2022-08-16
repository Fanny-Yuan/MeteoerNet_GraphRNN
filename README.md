## Dynamic point cloud processing for action recognition

This project adapts MeteorNet and Graph-RNN to human action recognition. The structure of the codes are shown below:

├─action_cls\
│  └─log\
│      ├─test\
│      └─train\
├─chain_interp_flow_preprocess\
├─Depth\
├─tf_ops\
│  ├─3d_interpolation\
│  ├─grouping\
│  └─sampling\
└─utils

Models are put in folder action_cls. The core part of models, such as the Meteor module and Graph-RNN module are put in utils. The dataset should be put in Depth. Chain_interp_flow_preprocess is the grouping code for MeteroNet, and tf_ops are codes for TF operators.

## Environment
These codes are running on:\
Python 3.5\
Tensorflow-gpu 1.9.0\
CUDA 11.6.1\
OpenCV

## Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them first by `make` under each ops subfolder. Make sure to change the CUDA_HOME as the same as yours. Change the arch version according to your CUDA. For example, our cuda Tesla V100, the corresponding arch version is sm_70. You can check <a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported">CUDA Compute Capability</a> to find the version that suits your GPU.

## Data processing

Download raw MSRAction3D dataset from <a href="https://drive.google.com/file/d/1djwAK3oZTAIFbCz531eClxINmsZgGO_H/view?usp=sharing">here</a> (~62MB). Extract the `.zip` file to get `Depth.rar` file and extract it in `Depth` directory. Then in this directory, run the following command to preprocess the data. `--num_cpu` flag is used to specify the number of CPUs to use during parallel processing.

```
python preprocess_file.py --input_dir /path/to/Depth --output_dir processed_data --num_cpu 11
```

## Training

The script for training and testing the model is `command_train.sh`. To train, use the following command.

```
sh command_train.sh
```

One may change the flags such as `num_frame`, `num_point` etc for different architecture specs.

If you want to train MeteorNet, change `model` to `model_cls_direct`;

If you want to train Graph-RNN, change `model` to `model_cls_graphrnn_v2`.



