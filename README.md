# DeformableVit
Use a deformable transformer to do the classification on the cifar-100 dataset.

## How to run the code
1. Install MSDAttention
  ```
    cd models/ops
    python setup.py build install
  ```
If you install it successfully, some files will appear under the ops directory.

2. Train model
```
  python Train.py
```
## Result
On test dataset:
![Epoch_0_Batch_0](https://github.com/Zhong-Zi-Zeng/DeformableVit/assets/102845636/477d6310-205c-4c24-a3af-987cde77584d)

On training dataset:
![tensorboard](https://github.com/Zhong-Zi-Zeng/DeformableVit/assets/102845636/c67855b1-9ee3-4f29-bd23-ffb1462eb51d)


## Acknowledgement
[Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR/tree/main)
