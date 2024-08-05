Assuming in DRAGON, one can run:

```
pip install torchtune
tune run --nproc_per_node <num_gpus> first_lora_finetune_distributed.py --config ./configs/8B_lora.yaml
```
To run lora finetune with `<num_gpus>` available.

If not in DRAGON, modify the paths to the model checkpoints in `./configs/8B_lora.yaml` accordingly.
