# 2022 CVPR Hyperbolic Vision Transformers: Combining Improvements in Metric Learning

[[Paper]](https://ieeexplore.ieee.org/document/9880306)
[[Code]](https://github.com/htdt/hyp_metric)

# Performance

| Method\Dataset           |   cifar   |  nuswide  |  flickr   |   coco    |
|:-------------------------|:---------:|:---------:|:---------:|:---------:|
| vit-32bit-hyperbolic     | 0.810@089 | 0.845@144 | 0.841@059 | 0.689@159 |
| vit-32bit-no_hyperbolic  | 0.955@44  | 0.866@14  | 0.845@14  | 0.719@49  |
| vit-128bit-hyperbolic    | 0.939@59  | 0.885@19  | 0.876@69  | 0.766@59  |
| vit-128bit-no-hyperbolic | 0.969@59  | 0.895@04  | 0.869@29  | 0.778@24  |
| resnet-32bit             | 0.677@084 | 0.806@054 | 0.798@054 | 0.634@114 |

# Parameters

```
# same
args.batch_size = 128
```

| Method\Type | backbone | n-epochs | n-bits |  opt  | scheduler |  lr  |  wd  | hyp_c |
|:-----------:|:--------:|:--------:|:------:|:-----:|:---------:|:----:|:----:|:-----:|
|     m1      |  vit-s   |   200    |   32   | adamw |   none    | 3e-5 | 0.02 |  0.1  |
|     m2      |  vit-s   |   200    |   32   | adamw |   none    | 3e-5 | 0.02 |   0   |
|   m1-128    |  vit-s   |   200    |  128   | adamw |   none    | 3e-5 | 0.02 |  0.1  |
|   m2-128    |  vit-s   |   200    |  128   | adamw |   none    | 3e-5 | 0.02 |   0   |
|     m5      |  resnet  |   200    |   32   | adamw |   none    | 3e-5 | 0.02 |  0.1  |

# Changes

1. lack of apex, see: https://github.com/NVIDIA/apex
2. ignore num_samples of sampler
3. smaller batch_size, due to the limited gpu memory
4. where is delta? fixed value (maybe hyp_c)
5. fix bug: skip_head -> skip_poincare
