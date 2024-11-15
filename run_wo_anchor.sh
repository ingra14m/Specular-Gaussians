# For mip-360 dataset
python train.py -s data/mipnerf-360/bicycle -m outputs/mip360/bicycle --eval -r 4 --is_real --asg_degree 12
python train.py -s data/mipnerf-360/bonsai -m outputs/mip360/bonsai --eval -r 2 --is_real --is_indoor --asg_degree 12
python train.py -s data/mipnerf-360/counter -m outputs/mip360/counter --eval -r 2 --is_real --is_indoor --asg_degree 12
python train.py -s data/mipnerf-360/flowers -m outputs/mip360/flowers --eval -r 4 --is_real --asg_degree 12
python train.py -s data/mipnerf-360/garden -m outputs/mip360/garden --eval -r 4 --is_real --asg_degree 12
python train.py -s data/mipnerf-360/kitchen -m outputs/mip360/kitchen --eval -r 2 --is_real --is_indoor --asg_degree 12
python train.py -s data/mipnerf-360/room -m outputs/mip360/room --eval -r 2 --is_real --is_indoor --asg_degree 12
python train.py -s data/mipnerf-360/stump -m outputs/mip360/stump --eval -r 4 --is_real --asg_degree 12
python train.py -s data/mipnerf-360/treehill -m outputs/mip360/treehill --eval -r 4 --is_real --asg_degree 12


# For nerf_synthetic dataset
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/chair -m outputs/blender/chair --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/drums -m outputs/blender/drums --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/ficus -m outputs/blender/ficus --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/hotdog -m outputs/blender/hotdog --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/lego -m outputs/blender/lego --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/materials -m outputs/blender/materials --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/mic -m outputs/blender/mic --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/ship -m outputs/blender/ship --eval

# For nsvf_synthetic dataset
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Bike/ -m outputs/nsvf/Bike --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Lifestyle/ -m outputs/nsvf/Lifestyle --eval -w
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Palace/ -m outputs/nsvf/Palace --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Robot/ -m outputs/nsvf/Robot --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Steamtrain/ -m outputs/nsvf/Steamtrain --eval -w
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Spaceship/ -m outputs/nsvf/Spaceship --eval -w
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Toad/ -m outputs/nsvf/Toad --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Wineholder/ -m outputs/nsvf/Wineholder --eval

# For our anisotropic dataset
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/ashtray -m outputs/asg/ashtray --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/dishes -m outputs/asg/dishes --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/headphone -m outputs/asg/headphone --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/jupyter -m outputs/asg/jupyter --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/lock -m outputs/asg/lock --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/plane -m outputs/asg/plane --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/record -m outputs/asg/record --eval
python train.py -s /media/data_nix/yzy/Git_Project/data/asg/teapot -m outputs/asg/teapot --eval
