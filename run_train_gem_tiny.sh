export DETECTRON2_DATASETS=./datasets/GSD-S-D2
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_net.py --num-gpus 4 \
        --config-file ./configs/gsd-s/semantic-segmentation/gem_sam_tiny_bs32_iter1w_steplr.yaml \
        MODEL.WEIGHTS ./pretrained_models/mobile_sam.pkl

