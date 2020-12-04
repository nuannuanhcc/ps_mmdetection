export DIR_NAME="reppoints"
python3 ./tools/test.py ./am_configs/reppoints.py ./work_dirs/$DIR_NAME/latest.pth --work-dir './work_dirs/'$DIR_NAME'' --eval bbox
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=295020 ./tools/dist_test.sh ./am_configs/reppoints.py ./work_dirs/$DIR_NAME/latest.pth 8 --work-dir './work_dirs/'$DIR_NAME'' --eval bbox
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=295021 ./tools/dist_test.sh ./am_configs/reppoints.py ./work_dirs/$DIR_NAME/latest.pth 4 --work-dir './work_dirs/'$DIR_NAME'' --eval bbox
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=295022 ./tools/dist_test.sh ./am_configs/reppoints.py ./work_dirs/$DIR_NAME/latest.pth 4 --work-dir './work_dirs/'$DIR_NAME'' --eval bbox
