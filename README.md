# Anti spoofing with the Datasouls dataset
![](https://habrastorage.org/webt/uv/7u/ws/uv7uwsjkcz732_vhf0opx3zfjrc.jpeg)

## Dataset

[ID & RD anti spoofing challenge](https://ods.ai/competitions/idrnd-facial-antispoofing)

Four types of images:
* real
* replay
* printed
* mask2d

## Training

### Define the config.
Example at [datasoluls_antispoof/configs](datasouls_antispoof/configs)

### Define the environmental variable `IMAGE_PATH` that points to the folder with the dataset.
Example:
```bash
export IMAGE_PATH=<path to the folder with images>
```
### Inference

```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpu> datasouls_antispoof/inference.py \
                                   -i <path to images> \
                                   -c <path to config> \
                                   -w <path to weights> \
                                   -o <output-path> \
                                   --fp16
```

### Pre-trained models

| Models        | Validation accuracy | Config file  | Weights |
| ------------- |:--------------------:| :------------:| :------: |
| swsl_resnext50_32x4d | 0.9673 | [Link](datasouls_antispoof/configs/2020-11-30b.yaml) | [Link](https://github.com/ternaus/datasouls_antispoof/releases/download/0.0.1/2020-11-30b_resnext50_32x4d.zip) |
