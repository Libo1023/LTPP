# RE-GCN Training, Validation, and Test 

cd backbone3_regcn/
mkdir models 
cd src/

# ICEWS14 
python get_history.py --dataset ICEWS14 
python main.py --dataset ICEWS14 --train-history-len 12 --test-history-len 12 --n-layers 2 --angle 10 --dropout 0.2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --add-static-graph --gpu 0 --n-epochs 50 

# ICEWS18 
python get_history.py --dataset ICEWS18 
python main.py --dataset ICEWS18 --train-history-len 12 --test-history-len 12 --n-layers 2 --angle 10 --dropout 0.2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --add-static-graph --gpu 0 --n-epochs 40 

# ICEWS05-15 
python get_history.py --dataset ICEWS05-15 
python main.py --dataset ICEWS05-15 --train-history-len 12 --test-history-len 12 --n-layers 2 --angle 10 --dropout 0.2 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --entity-prediction --add-static-graph --gpu 0 --n-epochs 25 

# RE-GCN + LTPP 

cd ../ 

# ICEWS14 
python ltpp_3_regcn.py --dataset ICEWS14 --gpu 0 --longtail-threshold 20 

# ICEWS18 
python ltpp_3_regcn.py --dataset ICEWS18 --gpu 0 --longtail-threshold 20 

# ICEWS05-15 
python ltpp_3_regcn.py --dataset ICEWS05-15 --gpu 0 --longtail-threshold 20 
