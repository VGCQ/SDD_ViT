<div align="center">
  <div>
  <h1>Sparse Double Descent in Vision Transformers: real or phantom threat?</h1> 

[![Static Badge](https://img.shields.io/badge/SDD_ViT-arXiv-red)](https://arxiv.org/abs/2307.14253)
[![Static Badge](https://img.shields.io/badge/SDD_ViT-Springer-blue)](https://link.springer.com/chapter/10.1007/978-3-031-43153-1_41)

  </div>

</div>

<div align="center">

<div>
    <a href='' target='_blank'>Victor Quétu</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Marta Milovanovic</a><sup>1</sup>&emsp;
    <a href='' target='_blank'>Enzo Tartaglione</a><sup>1</sup>&emsp;  
</div>
<div>
<sup>1</sup>LTCI, Télécom Paris, Institut Polytechnique de Paris&emsp;  

</div>
</div> <br>

📣 Published as a conference paper at ICIAP 2023.  <br>

This GitHub implements the key experiments of the following paper : [Sparse Double Descent in Vision Transformers: real or phantom threat?](https://arxiv.org/abs/2307.14253).

## Occurrence of Sparse Double Descent in Vision Transformers?

![teaser](images/SDD.png)

Figure: Test accuracy of ViT on **(Left.)** CIFAR-10 and **(Right.)** CIFAR-100 with different amount of label noise $\varepsilon$.

## Libraries
* Python = 3.10
* PyTorch = 1.13
* Torchvision = 0.14
* Numpy = 1.23

## Usage

In practice, you can begin with a set of defaults and optionally modify individual hyperparameters as desired. To view the hyperparameters for each subcommand, use the following command. 
```
main.py [subcommand] [...] --help
```

## Example Runs

To run a ViT on CIFAR-10 with 10% of label noise, batch size of 512, learning rate of 1e-4, weight decay of 0.03 for 200 epochs:
```python main.py```

To run a ResNet-18 on CIFAR-100 with 20% of label noise, batch size of 128, learning rate of 0.1, and weight decay of 1e-4 for 160 epochs:
```python main.py --model='resnet-18' --num_classes=100 --amount_noise=0.2 --batch_size=128 --learning_rate=0.1 --weight_decay=1e-4 --epochs=160```

## Citation
If you find this useful for your research, please cite the following paper.
```
@inproceedings{quetu2023sparse,
  title={Sparse Double Descent in Vision Transformers: real or phantom threat?},
  author={Qu{\'e}tu, Victor and Milovanovi{\'c}, Marta and Tartaglione, Enzo},
  booktitle={International Conference on Image Analysis and Processing},
  pages={490--502},
  year={2023},
  organization={Springer}
}

```
