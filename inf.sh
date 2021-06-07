export CUDA_VISIBLE_DEVICES='1'
chime4="/mnt/Data/user_vol_2/user_neillu/CHiME3/data/audio/16kHz/isolated/"
diffwave="/mnt/Data/user_vol_2/user_neillu/Diffwave/"
noisy_list="et05_bus_simu et05_caf_simu et05_ped_simu et05_str_simu et05_bus_real et05_caf_real et05_ped_real et05_str_real"
clean_list="et05_bth"


# stage 1: preparing data
# for x in $noisy_list; do
#     rm -r ${diffwave}/CHiME4/${x}_npy || true
# done
# for x in $noisy_list; do
#     echo "create npy under ${diffwave}/CHiME4/${x}_npy"
#     python src/diffwave/preprocess.py ${chime4}/${x} ${diffwave}/CHiME4/${x}_npy --se --test
# done

# stage 2: inference
for x in $noisy_list; do
    rm -r ${diffwave}/out/CHiME4/${x} || true
done
for x in $noisy_list; do
    mkdir -p ${diffwave}/out/vocoder/model413888/CHiME4/${x}
    echo "inference enhanced wav file under ${diffwave}/out/model413888/CHiME4/${x}"
    python src/diffwave/inference.py  ${diffwave}/model_vocoder/weights-413888.pt ${diffwave}/CHiME4/${x}_npy -o ${diffwave}/out/vocoder/model413888/CHiME4/${x} # --se
done



