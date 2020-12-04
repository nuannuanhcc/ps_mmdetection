python3 ./tools/train.py ./am_configs/reppoints.py --work-dir './work_dirs/reppoints'
#python3 ./tools/train.py ./am_configs/fcos.py --work-dir './work_dirs/fcos'
#python3 ./tools/train.py ./am_configs/retinanet.py --work-dir './work_dirs/retinanet'
#python3 ./tools/train.py ./am_configs/atss.py --work-dir './work_dirs/atss'

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=295000 ./tools/dist_train.sh ./am_configs/reppoints.py 8 --work-dir './work_dirs/reppoints'
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=295001 ./tools/dist_train.sh ./am_configs/reppoints.py 4 --work-dir './work_dirs/reppoints'
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=295002 ./tools/dist_train.sh ./am_configs/reppoints.py 4 --work-dir './work_dirs/reppoints'
