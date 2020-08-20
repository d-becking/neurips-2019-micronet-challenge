# NeurIPS 2019 MicroNet Challenge

### We encourage our readers to take a look at the latest [improvements](https://github.com/d-becking/efficientCNNs)

### You may also want to read the arXiv.org preprint: https://arxiv.org/abs/2004.01077 

If you find this code useful in your research, please cite:
```
@InProceedings{Marban_2020_EC2T,
author = {Marban, Arturo and Becking, Daniel and Wiedemann, Simon and Samek, Wojciech},
title = {Learning Sparse & Ternary Neural Networks With Entropy-Constrained Trained Ternarization (EC2T)},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020},
pages={3105-3113},
doi = {10.1109/CVPRW50498.2020.00369}
}
```

In this project two algorithms are implemented: first, an algorithm which compresses neural networks by applying an 
entropy controlled ternary quantization and, second, a compound scaling approach which finds efficient architectures 
in terms of scaling model width and depth. 

The resulting neural networks solve the NeurIPS MicroNet Challenge's target tasks to a specified quality level:

https://micronet-challenge.github.io/

Further information on the compound scaling and quantization concept can be found in the _writeup.pdf_ file in this git 
repository.

### Folder structure:

* **model package**: Here the MicroNet and EfficientNet model builders can be found. Not executing tasks.
* **model_compound_scaling package**: Here a grid search on width and depth scaling factors is applied to the MicroNet.
* **model_quantization package**: Here the MicroNet found by the compound scaling approach (or an existing 
EfficientNet) are quantized ternary.
To make the model as sparse as possible a grid search on lambda dividers is applied with lambda controlling the 
intensity of the entropy constraint which is applied to the weights to push them into the zero valued cluster (cluster 
of lowest information content). 
* **model_scoring package**: Here code can be found which calculates the number of scoring parameters and math 
operations for inference as declined on the challenge's website. 
Furthermore a summary of the model statistics is printed.

### Scores:
**Counting additions and multiplications as FP16 ops**

|    *Task*   |*# Params*|*Top-1 Acc.*|*Sparsity*|*# Scoring params*|*# Inference FLOP*|*Overall Score*|
|:-----------:|:--------:|:----------:|:--------:|:----------------:|:-----------------:|:-------------:|
| `CIFAR-100` |   8.1M   |    80.13%  |   90.49% |      0.43M       |       66.77M      |    0.0182     |
| `ImageNet`  |   7.8M   |    75.03%  |   46.33% |      1.33M       |       250.31M     |    0.4070     |


**Counting additions as FP32 ops and multiplications as FP16 ops**

|    *Task*   |*# Params*|*Top-1 Acc.*|*Sparsity*|*# Scoring params*|*# Inference FLOP*|*Overall Score*|
|:-----------:|:--------:|:----------:|:--------:|:----------------:|:-----------------:|:-------------:|
| `CIFAR-100` |   8.1M   |    80.13%  |   90.49% |      0.43M       |       129.83M     |    0.0242     |
| `ImageNet`  |   7.8M   |    75.03%  |   46.33% |      1.33M       |       455.19M     |    0.5821     |

### Code execution:

To execute compound scaling, quantization or scoring run the according python run files at the top level of this project, 
e.g. **python -u run_compound_scaling.py**. Hyperparameters can be altered via the parser. To get all parser arguments
execute 
```
python -u run_compound_scaling.py --help
python -u run_quantization.py --help
python -u run_scoring.py --help
```
The **--help** flag will list all (optional) arguments for each run file. As an example
```
python -u run_scoring.py --no-halfprecision --t-model best_ternary_imagenet_micronet.pt --image-size 200 
--no-cuda > console.txt
```
executes the scoring procedure with full precision parameters (32 bit), the quantized network which solved
ImageNet best, with an input image size of 200x200 px and a CPU mapping instead of GPU usage. The console output can 
be saved optionally in a text file by appending **> console.txt**.

For using multiple GPUs set the CUDA_VISIBLE_DEVICES environment variable before executing the packages
```
export CUDA_VISIBLE_DEVICES=0,1
```

### Full example to reproduce the results of the CIFAR-100 task:
```
python -u run_compound_scaling.py --epochs 250 --batch-size 128 --grid 1.4 1.2 --phi 3.5 --dataset CIFAR100 
--image-size 32 
```
Copy the best full-precision model from `./model_compound_scaling/saved_models/best_model.pt` to
`./model_quantization/trained_fp_models` and optionally rename it, e.g. _MicroNet_d14_w12_phi35_acc8146_params8_06m.th_.
```
python -u run_quantization.py --model-dict MicroNet_d14_w12_phi35_acc8146_params8_06m.th --batch-size 128 --epochs 20 
--retrain-epochs 20 --ini-c-divrs 0.45 --lambda-max-divrs 0.15 --model cifar_micronet --dw-multps 1.4 1.2 --phi 3.5 
--dataset CIFAR100
```
The best quantized model can be found in `./model_quantization/saved_models/Ternary_best_acc.pt`.
Copy it to `./model_scoring/trained_t_models` and optionally rename it, e.g. 
_best_ternary_cifar_micronet.pt_.
Execute:
```
python -u run_scoring.py --t-model best_ternary_cifar_micronet.pt  --model cifar_micronet  
--dataset CIFAR100 --eval --halfprecision --add-bits 16
```

### Full example to reproduce the results of the ImageNet task:
Download the pretrained EfficientNet-B1 by Luke Melas-Kyriazi (https://github.com/lukemelas/EfficientNet-PyTorch) or
train it from scratch. In our environment EfficientNet-B1, which can be found in 
`./model_quantization/trained_fp_models`, achieves an ImageNet accuracy of 78.4% which is 0.4% less than described 
in the paper.

In a first step we quantize the "expand", "projection" and "head" convolutions of EfficientNet-B1 with our entropy 
controlled ternary approach:
```
python -u run_quantization.py --batch-size 128 --val-batch-size 256 --ini-c-divrs 0.2 --lambda-max-divrs 0.125 
--model efficientnet-b1 --model-dict efficientnet_b1.pt --dataset ImageNet --image-size 224 
--data-path [your_path] --epochs 20 --retrain-epochs 15
```
In a second step the non-ternary Squeeze-and-Excitation layers plus the fully connected layer run through the 
same algorithm but only the assignment to the zero cluster is executed (entropy controlled pruning, important: set 
--do-prune flag). Copy the best model from `./model_quantization/saved_models/` to 
`./model_quantization/trained_fp_models` and optionally rename it, e.g. _B1_ternary_exp_proj_head.pt_. As the non-zero 
values remain in full precision, hyperparameters _ini-c-divrs_ and _lambda-max-divrs_ can be increased:
```
python -u run_quantization.py --batch-size 128 --val-batch-size 256 --ini-c-divrs 0.25 --lambda-max-divrs 0.15 
--model efficientnet-b1 --model-dict B1_ternary_exp_proj_head.pt --dataset ImageNet --image-size 224 
--data-path [your_path] --epochs 10 --retrain-epochs 0 --do-prune --no-resume
```
Copy the best model from `./model_quantization/saved_models/` to `./model_scoring/trained_t_models` and optionally 
rename it, e.g. _best_ternary_and_pruned_imagenet_efficientnet.pt_.
Execute:
```
python -u run_scoring.py --dataset ImageNet --data-path [your_path] --val-batch-size 256 
--image-size 256 --model efficientnet-b1 --t-model best_ternary_and_pruned_imagenet_efficientnet.pt 
--prune-threshold 0.01 --eval --halfprecision --add-bits 16
```
