chime4="/mnt/Data/user_vol_2/user_neillu/CHiME3/data/audio/16kHz/isolated/"
diffwave="/mnt/Data/user_vol_2/user_neillu/Diffwave/"
noisy_list="tr05_bus_simu tr05_caf_simu tr05_ped_simu tr05_str_simu"


#stage 1: preparing data
for x in $noisy_list; do
    rm -r ${diffwave}/CHiME4/${x}_npy || true
    echo "create npy under ${diffwave}/CHiME4/${x}_npy"
    python -m diffwave.preprocess ${chime4}/${x} ${diffwave}/CHiME4/${x}_npy
done

noisy_path_list=""
for x in $noisy_list; do
    noisy_path_list="${noisy_path_list} ${diffwave}/CHiME4/${x}_npy"
done

#stage 2: training model
python -m diffwave ${diffwave}/model_se ${chime4}/tr05_org/  [${noisy_path_list}]


