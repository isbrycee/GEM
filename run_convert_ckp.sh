export DETECTRON2_DATASETS=./datasets/GSD-S-D2

export PATH=/home/ssd5/wangyunhao02/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/home/ssd5/wangyunhao02/cuda-10.2/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ssd5/wangyunhao02/cudnn-8.0/cuda/lib64:$LD_LIBRARY_PATH
# export PATH=/opt/compiler/gcc-4.8.2/bin:/opt/compiler/gcc-4.8.2/lib64:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

python tools/convert-sam-tiny-to-d2.py xxx/MobileSAM-master/weights/mobile_sam.pt xxx/MobileSAM-master/weights/mobile_sam.pkl