export CUDA_VISIBLE_DEVICES='4'
 
stage=2
ckp="607452"
task="se" #"vocoder" or "se"
model_name="model_se_full_fix_in_unfix"
fix="--fix_in"

. ./path.sh

esp_path=/mnt/Data/user_vol_2/user_neillu/ESPNet3/espnet/egs2/chime4/enh1/
test_list="dt05_simu_isolated_1ch_track et05_simu_isolated_1ch_track"


if [[ "$task" == "se" ]]; then
    wav_root=${chime4_noisy}
    spec_root=${diffwave}/spec/Test_Noisy
    spec_type="noisy spectrum"
fi
# elif [[ "$task" == "vocoder" ]]; then
#     wav_root=${chime4_clean}
#     spec_root=${diffwave}/spec/Test_Clean
#     spec_type="clean Mel-spectrum"
# fi


if [ ${stage} -le 1 ]; then
    echo "stage 1 : preparing testing data"

    for x in $test_list; do
        wav_scp=${esp_path}/dump/raw/${x}/wav.scp
        spec_path=${spec_root}/${x} 
        spec_scp=${spec_root}/${x}/spec.scp
        echo "create ${spec_type} from ${wav_scp} to ${spec_path}"
        rm -r ${spec_path} 2>/dev/null
        mkdir -p ${spec_path}
        python src/diffwave/preprocess_scp.py ${wav_scp} ${esp_path} ${spec_scp} ${spec_path} --${task} --test 
    done
fi

if [ ${stage} -le 2 ]; then
    #"exp/enh_train_enh_conv_tasnet_raw/enhanced_dt05_simu_isolated_1ch_track/spk1.scp"
    #"exp/enh_train_enh_conv_tasnet_raw/enhanced_et05_simu_isolated_1ch_track/spk1.scp"
    enhanced_path=${diffwave}/Enhanced/${model_name}/model${ckp}_fast_noisy/
    rm -r ${enhanced_path} 2>/dev/null
    echo "stage 2 : Generate spk1.scp for test set and dev set"
    for x in $test_list; do
        wav_scp=${esp_path}/dump/raw/${x}/wav.scp
        spec_scp=${spec_root}/${x}/spec.scp
        outwav_scp=${enhanced_path}/${x}/spk1.scp
        python src/diffwave/inference_scp.py --fast ${diffwave}/${model_name}/weights-${ckp}.pt ${wav_scp} ${esp_path} ${spec_scp} ${outwav_scp} -o ${enhanced_path} ${fix} --${task}
    done
fi

# if [ ${stage} -le 3 ]; then
#     #"exp/enh_train_enh_conv_tasnet_raw/enhanced_dt05_simu_isolated_1ch_track/spk1.scp"
#     #"exp/enh_train_enh_conv_tasnet_raw/enhanced_et05_simu_isolated_1ch_track/spk1.scp"
#     enhanced_path=${diffwave}/Enhanced/${model_name}/model${ckp}/
#     echo "stage 3 : Cut short the spk1.scp as long as the wav.scp"
#     rm -r ${enhanced_path}/align
#     for x in $test_list; do
#         wav_scp=${esp_path}/dump/raw/${x}/wav.scp
#         outwav_scp=${enhanced_path}/${x}/spk1.scp
#         python src/diffwave/cutshort_scp.py ${wav_scp} ${esp_path} ${outwav_scp} -o ${enhanced_path}/align 
#     done
# fi



