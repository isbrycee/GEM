export DETECTRON2_DATASETS=./datasets/GSD-S-D2
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_net.py --num-gpus 4 \
        --config-file ./configs/gsd-s/semantic-segmentation/gem_sam_base_bs16_iter2w_steplr.yaml \
        MODEL.WEIGHTS ./pretrained_models/sam_vit_b_01ec64_wopos.pkl
