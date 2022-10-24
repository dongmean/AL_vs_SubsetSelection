# Active Learning is a Strong Baseline for Data Subset Selection (NeurIPS 2022, [HITY Workshop](https://hity-workshop.github.io/NeurIPS2022/))

by [Dongmin Park](https://scholar.google.com/citations?user=4xXYQl0AAAAJ&hl=ko)<sup>1</sup>, [Dimitris Papailiopoulos](https://scholar.google.com/citations?user=hYi6i9sAAAAJ&hl=ko)<sup>2</sup>, [Kangwook Lee](https://scholar.google.com/citations?user=sCEl8r-n5VEC&hl=ko&oi=ao)<sup>2</sup>.

<sup>1</sup> KAIST, <sup>2</sup> University of Wisconsin-Madison

* **`Oct 21, 2022`:** **Our work is accepted to HITY Workshop at NeurIPS 2022.**
* **To be published soon.**

# How to run

### OUR Active Learning Baseline

Go to the AL/ folder

* CIFAR10
```bash
python3 main.py --gpu 0 --data_path=$your_data_folder --dataset 'CIFAR10' --n-class 10 --model 'ResNet18' \
                        --method 'Uncertainty' --uncertainty 'Margin' --n-query 1000 --epochs 200 --batch-size 128
```
* CIFAR100
```bash
python3 main.py --gpu 0 --data_path=$your_data_folder --dataset 'CIFAR100' --n-class 100 --model 'ResNet18' \
                        --method 'Uncertainty' --uncertainty 'Margin' --n-query 1000 --epochs 200 --batch-size 128
```
* ImageNet30
```bash
python3 main.py --gpu 0 --data_path=$your_data_folder --dataset 'ImageNet30' --n-class 30 --model 'ResNet18' \
                        --method 'Uncertainty' --uncertainty 'Margin' --n-query 780 --epochs 200 --batch-size 128
```

### Existing Subset Selection
###: Uniform, Uncertainty (Margin), Forgetting, GraNd, kCenterGreedy, GraphCut, Glister, etc

Go to the DeepCore/ folder

* CIFAR10, CIFAR100, ImageNet50
```bash
python3 main.py --data_path=$your_data_folder --datset $dataset --model $arch --selection $selection_algorithm --fraction $target_fraction
```

# Requirements

```
torch: +1.3.0
torchvision: 1.7.0
prefetch_generator: 1.0.1
submodlib: 1.1.5
diffdist: 0.1
scikit-learn: 0.24.2
scipy: 1.5.4
ptflops: 0.6.9
```

# Citation

To be available soon

# References

* DeepCore library \[[code](https://github.com/PatrickZH/DeepCore)\] : DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning, Guo et al. 2022.
* AL library \[[code](https://github.com/kaist-dmlab/MQNet)\] : Meta-Query-Net: Resolving Purity-Informativeness Dilemma in Open-set Active Learning, Park et al. NeurIPS 2022.
