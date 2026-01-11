# RE-NET Training, Validation, and Test 

cd backbone2_renet/
cd reproduce_commands/

# ICEWS14 
# Step 1: Data Preprocessing 
python renet_reproduce.py preprocess -d ICEWS14 --data-dir ./data
# Step 2: Pretrain Global Model
python renet_reproduce.py pretrain -d ICEWS14 --gpu 0 --n-hidden 200 --seq-len 12 --dropout 0.5 --lr 1e-3 --max-epochs 100 --batch-size 256
# Step 3: Train and Validate RE-NET Model
python renet_reproduce.py train -d ICEWS14 --gpu 0 --n-hidden 200 --seq-len 12 --dropout 0.5 --lr 1e-3 --max-epochs 20 --batch-size 256
# Step 4: Test RE-NET Model 
python renet_reproduce.py test -d ICEWS14 --gpu 0 --n-hidden 200 --seq-len 12 --dropout 0.5 --lr 1e-3

# ICEWS18 
python renet_reproduce.py preprocess -d ICEWS14 --data-dir ./data
python renet_reproduce.py pretrain -d ICEWS18 --gpu 0 --n-hidden 200 --seq-len 12 --dropout 0.5 --lr 1e-3 --max-epochs 100 --batch-size 256
python renet_reproduce.py train -d ICEWS18 --gpu 0 --n-hidden 200 --seq-len 12 --dropout 0.5 --lr 1e-3 --max-epochs 20 --batch-size 256
python renet_reproduce.py test -d ICEWS18 --gpu 0 --n-hidden 200 --seq-len 12 --dropout 0.5 --lr 1e-3 

# ICEWS05-15 
python renet_reproduce.py preprocess -d ICEWS14 --data-dir ./data
python renet_reproduce.py pretrain -d ICEWS05-15 --gpu 0 --n-hidden 100 --seq-len 7 --dropout 0.5 --lr 1e-3 --max-epochs 100 --batch-size 256
python renet_reproduce.py train -d ICEWS05-15 --gpu 0 --n-hidden 100 --seq-len 7 --dropout 0.5 --lr 1e-3 --max-epochs 10 --batch-size 256 
python renet_reproduce.py test -d ICEWS05-15 --gpu 0 --n-hidden 100 --seq-len 7 --dropout 0.5 --lr 1e-3

# RE-NET + LTPP 

cd ../ 

# ICEWS14 
python ltpp_2_renet.py --dataset ICEWS14 --gpu 0 --longtail-threshold 20 

# ICEWS18 
python ltpp_2_renet.py --dataset ICEWS18 --gpu 0 --longtail-threshold 20 

# ICEWS05-15 
python ltpp_2_renet.py --dataset ICEWS05-15 --gpu 0 --longtail-threshold 20 