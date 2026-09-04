export CUDA_VISIBLE_DEVICES=5
python inference.py --dataset isaid --data-root /home/user9/dataset/user9/INSID_data --exp-name insid3-isaid --crf-mask-refinement --fold 0



python inference.py --dataset isaid --data-root /home/user9/dataset/user9/INSID_data --exp-name insid3-isaid --crf-mask-refinement --fold 1



python inference.py --dataset isaid --data-root /home/user9/dataset/user9/INSID_data --exp-name insid3-isaid --crf-mask-refinement --fold 2


# bash scripts/isaid.sh