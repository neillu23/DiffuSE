export CUDA_VISIBLE_DEVICES='0,2'
# chime4="/home/iis-cvl/DW/CHIME4/"
chime4_noisy="/home/iis-cvl/DW/nn-gev/data/audio/16kHz/isolated/"
chime4_clean="/home/iis-cvl/DW/nn-gev/data/audio/16kHz/isolated_ext/"
diffwave="/home/iis-cvl/DW/out"
noisy_list="tr05_bus_simu tr05_caf_simu tr05_ped_simu tr05_str_simu"
clean_list="tr05_org"


# #stage 1: preparing data
# for x in $noisy_list; do
#     rm -r ${diffwave}/CHiME4_full/${x} || true
#     mkdir -p ${diffwave}/CHiME4_full/${x}
#     echo "create npy under ${diffwave}/CHiME4_full/${x}"
#     python src/diffwave/preprocess.py ${chime4_noisy}/${x} ${diffwave}/CHiME4_full/${x} --se
# done

noisy_path_list=""
for x in $noisy_list; do
    noisy_path_list="${noisy_path_list} ${diffwave}/spec/CHiME4_Noisy/${x}"
done

#stage 2: training model
python src/diffwave/__main__.py ${diffwave}/model_se_full ${chime4_clean}/   ${noisy_path_list}  --se


