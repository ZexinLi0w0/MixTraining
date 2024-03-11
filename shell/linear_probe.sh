CUDA_VISIBLE_DEVICES=1 python resnet_train_standard_ae.py --k 1 --mix_ratio 0.0 --output_model_path 'test.pt' --dataset 'cifar10' --ft_epoch 100 --pt_epoch 0 --from_pretrain --linear_probe
