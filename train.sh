export CUDA_VISIBLE_DEVICES='4,7'
chime4="/mnt/Data/user_vol_2/user_neillu/CHiME3/data/audio/16kHz/isolated/"
diffwave="/mnt/Data/user_vol_2/user_neillu/Diffwave/"
noisy_list="tr05_bus_simu tr05_caf_simu tr05_ped_simu tr05_str_simu"
clean_list="tr05_org"


# #stage 1: preparing data
# for x in $noisy_list; do
#     rm -r ${diffwave}/CHiME4/${x}_npy || true
#     echo "create npy under ${diffwave}/CHiME4/${x}_npy"
#     python src/diffwave/preprocess.py ${chime4}/${x} ${diffwave}/CHiME4/${x}_npy --se
# done

 noisy_path_list=""
 for x in $noisy_list; do
     noisy_path_list="${noisy_path_list} ${diffwave}/CHiME4/${x}_npy"
 done

#stage 2: training model
python src/diffwave/__main__.py ${diffwave}/model_se_fix ${chime4}/tr05_org/   ${noisy_path_list}  --se --fix --pretrain ${diffwave}/model_vocoder/weights-765336.pt 


