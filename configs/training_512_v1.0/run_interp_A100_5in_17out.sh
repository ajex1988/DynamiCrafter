# NCCL configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_TOPO_FILE=/tmp/topo.txt

# args
name="training_512_v1.0"
config_file=configs/${name}/config_interp_youhq_5in_17out.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/workspace/shared-dir/zzhu/tmp/20250227"

mkdir -p $save_root/${name}_interp

## run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
--nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name ${name}_interp \
--logdir $save_root \
--devices 4 \
lightning.trainer.num_nodes=1

## debugging
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 -m torch.distributed.launch \
# --nproc_per_node=6 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
# ./main/trainer.py \
# --base $config_file \
# --train \
# --name ${name}_interp \
# --logdir $save_root \
# --devices 6 \
# lightning.trainer.num_nodes=1