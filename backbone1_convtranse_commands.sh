# ConvTransE Training, Validation, and Test 

cd backbone1_convtranse/
mkdir models
cd src/

# ICEWS14 
python get_history.py --dataset ICEWS14 
python main.py --dataset ICEWS14 --train-history-len 12 --test-history-len 12 --lr 0.001 --n-hidden 200 --decoder timeconvtranse --layer-norm --input-dropout 0.2 --hidden-dropout 0.2 --feat-dropout 0.2 --evaluate-every 1 --gpu 0 --save decoder_only --n-epochs 50 

# ICEWS18 
python get_history.py --dataset ICEWS18 
python main.py --dataset ICEWS18 --train-history-len 12 --test-history-len 12 --lr 0.001 --n-hidden 200 --decoder timeconvtranse --layer-norm --input-dropout 0.2 --hidden-dropout 0.2 --feat-dropout 0.2 --evaluate-every 1 --gpu 0 --save decoder_only --n-epochs 35 

# ICEWS05-15 
python get_history.py --dataset ICEWS05-15 
python main.py --dataset ICEWS05-15 --train-history-len 12 --test-history-len 12 --lr 0.001 --n-hidden 200 --decoder timeconvtranse --layer-norm --input-dropout 0.2 --hidden-dropout 0.2 --feat-dropout 0.2 --evaluate-every 1 --gpu 0 --save decoder_only --n-epochs 30 

# ConvTransE + LTPP 

cd ../ 

# ICEWS14 
python ltpp_1_convtranse.py --dataset ICEWS14 --gpu 0 --longtail-threshold 20 

# ICEWS18 
python ltpp_1_convtranse.py --dataset ICEWS18 --gpu 0 --longtail-threshold 20 

# ICEWS05-15 
python ltpp_1_convtranse.py --dataset ICEWS05-15 --gpu 0 --longtail-threshold 20 
