

export CUDA_VISIBLE_DEVICES=1
python inference.py --dataset paco_part --data-root /home/user9/dataset/user9/INSID_data/ --exp-name insid3-paco_part --crf-mask-refinement --fold 0


python inference.py --dataset paco_part --data-root /home/user9/dataset/user9/INSID_data/ --exp-name insid3-paco_part --crf-mask-refinement --fold 1


python inference.py --dataset paco_part --data-root /home/user9/dataset/user9/INSID_data/ --exp-name insid3-paco_part --crf-mask-refinement --fold 2


python inference.py --dataset paco_part --data-root /home/user9/dataset/user9/INSID_data/ --exp-name insid3-paco_part --crf-mask-refinement --fold 3



# bash scripts/paco.sh