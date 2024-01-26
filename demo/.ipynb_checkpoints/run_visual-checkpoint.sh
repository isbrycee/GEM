python demo.py --config-file ../configs/gsd-s/semantic-segmentation/gem_sam_tiny_bs32_iter1w_steplr.yaml \
  --input ../test_images/*.jpg \
  --output ../output \
  --opts MODEL.WEIGHTS model.pth \
  MODEL.DEVICE 'cpu'