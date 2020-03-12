source activate tf1_gpu
export CUDA_VISIBLE_DEVICES=1
nohup python main.py > ./log/1/save 2>&1 &
tailf ./log/1/save
