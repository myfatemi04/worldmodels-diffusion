scp pusht.zip vast:/root/pusht.zip

ssh vast << EOF

cd /root
apt install -y libgl1-mesa-glx htop zip unzip git-lfs ffmpeg python3.10-dev clang

unzip pusht.zip

git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git

cd cosmos-predict2.5
uv sync --extra=cu128
uv pip install accelerate diffusers av loguru matplotlib gymnasium gym-pusht "pymunk<7"

git lfs pull

ln -s /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so
ln -s /root/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 /root/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so

echo export HF_TOKEN=$HUGGINGFACE_API_KEY >> ~/.bashrc
echo export LD_LIBRARY_PATH=/root/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc

EOF

