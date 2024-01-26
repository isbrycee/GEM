export DETECTRON2_DATASETS=./datasets/GSD-S-D2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python train_net.py --eval-only --num-gpus 8 --config-file ./configs/gsd-s/semantic-segmentation/gem_sam_tiny_bs32_iter1w_steplr.yaml \
                MODEL.WEIGHTS model.pth \
