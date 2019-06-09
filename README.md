# Results

## experiment settings

We perform experiments on the widely used MNIST dataset.

To evaluate the models’ performance under adversarial attack, we use the FGSM and PGD attack method implemented in `cleverhans` library. 

Specific Settings:

```json
"FGSM":{"eps":0.3, "clip_min":0.0, "clip_max":1.0}
"PGD":{"eps":0.3, "eps_iter":0.01, "nb_iter":40, "clip_min":0.0, "clip_max":1.0}
```



## results:

| Model                                    | nat      | fgsm     | pgd      | c_gap                  |
| ---------------------------------------- | -------- | -------- | -------- | ---------------------- |
| L2NonExpa_resnet20-w:0.1-div_before_conv:False | 74.0500% | 73.9000% | 61.3900% | 9.103460385517793e-08  |
| L2NonExpa_resnet20-w:0.1-div_before_conv:True | 98.8300% | 1.1600%  | 0.3400%  | 1.6781422734260558     |
| L2NonExpa_resnet20-w:0.5-div_before_conv:False | 90.4400% | 90.2100% | 54.7900% | 2.2296798761090033e-06 |
| L2NonExpa_resnet20-w:0.5-div_before_conv:True | 93.9200% | 3.9900%  | 1.0700%  | 0.47031465768814085    |
| L2NonExpa_resnet20-w:0-div_before_conv:False | 66.9400% | 66.9100% | 57.8100% | 1.6424678284110425e-11 |
| L2NonExpa_resnet20-w:0-div_before_conv:True | 20.3700% | 20.3200% | 22.3300% | 1.1156944923484959e-09 |

| Model                                    | nat      | fgsm     | pgd      | c_gap                 |
| ---------------------------------------- | -------- | -------- | -------- | --------------------- |
| L2NonExpaConvNet-w:0.1-div_before_conv:False | 88.8300% | 88.4200% | 65.3500% | 0.003372525749728084  |
| L2NonExpaConvNet-w:0.1-div_before_conv:True | 99.0000% | 3.6600%  | 0.9100%  | 0.7210915863513945    |
| L2NonExpaConvNet-w:0.5-div_before_conv:False | 79.3700% | 77.0500% | 65.8500% | 0.0007947990030515938 |
| L2NonExpaConvNet-w:0.5-div_before_conv:True | 97.4200% | 4.3200%  | 3.1000%  | 0.5776241004467011    |
| L2NonExpaConvNet-w:0-div_before_conv:False | 86.8900% | 86.9000% | 86.1000% | 0.004795837821438908  |
| L2NonExpaConvNet-w:0-div_before_conv:True | 82.3000% | 82.2500% | 70.5900% | 0.005532516865059733  |

## Conclusion

- Our main contributions are:
  - We extend the L2NNN framework to modern neural network architectures like ResNet.
  - We extensively experimented with some design choices of  L2NNN.
- Our main findings are:
  - We find that L2-non-expansive ResNet can achieve larger confidence gap, thus provides more certified defense against white box adversarial attacks.
  - We find that although some design options of L2NNN (like adding weight regularization loss and divide kernel size before convolution)  can achieve better confidence gap (thus promoting certified defense),  downgrade the model’s performance under real attacks like FGSM and PGD. Since attacking algorithms give an upper bound of the model’s performance while confidence gaps give a lower bound, the tradeoff between these two bounds is an interesting problem.
