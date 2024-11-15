# For mip-360 dataset
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/bonsai -m outputs/mip360/bonsai-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/counter -m outputs/mip360/counter-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/kitchen -m outputs/mip360/kitchen-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/room -m outputs/mip360/room-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 2
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/bicycle -m outputs/mip360/bicycle-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/flowers -m outputs/mip360/flowers-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/garden -m outputs/mip360/garden-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/stump -m outputs/mip360/stump-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/mipnerf-360/treehill -m outputs/mip360/treehill-anchor --eval --voxel_size 0.001 --update_init_factor 16 --iterations 30_000 --use_c2f -r 4

# For nerf_synthetic dataset
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/chair -m outputs/blender/chair-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/drums -m outputs/blender/drums-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/ficus -m outputs/blender/ficus-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/hotdog -m outputs/blender/hotdog-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/lego -m outputs/blender/lego-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/materials -m outputs/blender/materials-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/mic -m outputs/blender/mic-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/nerf_synthetic/ship -m outputs/blender/ship-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000

# For nsvf_synthetic dataset
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Bike/ -m outputs/nsvf/Bike-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Lifestyle/ -m outputs/nsvf/Lifestyle-anchor --eval -w --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Palace/ -m outputs/nsvf/Palace-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Robot/ -m outputs/nsvf/Robot-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Steamtrain/ -m outputs/nsvf/Steamtrain-anchor --eval -w --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Spaceship/ -m outputs/nsvf/Spaceship-anchor --eval -w --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Toad/ -m outputs/nsvf/Toad-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/Synthetic_NSVF/Wineholder/ -m outputs/nsvf/Wineholder-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000

# For our anisotropic dataset
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/ashtray -m outputs/asg/ashtray-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/dishes -m outputs/asg/dishes-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/headphone -m outputs/asg/headphone-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/jupyter -m outputs/asg/jupyter-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/lock -m outputs/asg/lock-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/plane -m outputs/asg/plane-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/record -m outputs/asg/record-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000
python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/asg/teapot -m outputs/asg/teapot-anchor --eval --voxel_size 0.001 --update_init_factor 4 --iterations 30_000

# python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/tandt/train -m outputs/tandt/train --eval --voxel_size 0.01 --update_init_factor 16 --iterations 30_000
# python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/tandt/truck -m outputs/tandt/truck --eval --voxel_size 0.01 --update_init_factor 16 --iterations 30_000

# python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/db/drjohnson -m outputs/db/drjohnson --eval --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 --use_c2f
# python train_anchor.py -s /media/data_nix/yzy/Git_Project/data/tandt_db/db/playroom -m outputs/db/playroom --eval --voxel_size 0.005 --update_init_factor 16 --iterations 30_000 --use_c2f
