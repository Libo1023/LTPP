# TiRGN Training, Validation, and Test 

cd backbone4_tirgn/
mkdir models 
cd src/

# ICEWS14 
python get_history.py --dataset ICEWS14 
python main.py --dataset ICEWS14 --history-rate 0.3 --train-history-len 12 --test-history-len 12 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --n-epochs 50 

# ICEWS18 
python get_history.py --dataset ICEWS18 
python main.py --dataset ICEWS18 --history-rate 0.3 --train-history-len 12 --test-history-len 12 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --n-epochs 50 

# ICEWS05-15 
python get_history.py --dataset ICEWS05-15 
python main.py --dataset ICEWS05-15 --history-rate 0.3 --train-history-len 12 --test-history-len 12 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 14 --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint --n-epochs 35 

# TiRGN + LTPP 

cd ../ 

# ICEWS14 
python ltpp_4_tirgn.py --dataset ICEWS14 --gpu 0 --longtail-threshold 20 

# ICEWS18 
python ltpp_4_tirgn.py --dataset ICEWS18 --gpu 0 --longtail-threshold 20 

# ICEWS05-15 
python ltpp_4_tirgn.py --dataset ICEWS05-15 --gpu 0 --longtail-threshold 20 
