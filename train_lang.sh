DATE=$(date +'%Y%m%d-%H%M')
python train_lang.py 2>&1 | tee train_${DATE}.log
