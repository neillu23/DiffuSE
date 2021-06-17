export CUDA_VISIBLE_DEVICES='1'
 
stage=2
ckp="412104"
task="vocoder" #"vocoder" or "se"
model_name="model_vocoder_full"

. ./path.sh

chime4_noisy="${chime4}/isolated/"
chime4_clean="${chime4}/isolated_ext/"
test_list="et05_bus_simu et05_caf_simu et05_ped_simu et05_str_simu"
# noisy_list="et05_bus_simu et05_caf_simu et05_ped_simu et05_str_simu et05_bus_real et05_caf_real et05_ped_real et05_str_real"


if [[ "$task" == "se" ]]; then
    wav_root=${chime4_noisy}
    spec_root=${diffwave}/spec/CHiME4_Noisy
    spec_type="noisy spectrum"

elif [[ "$task" == "vocoder" ]]; then
    wav_root=${chime4_clean}
    spec_root=${diffwave}/spec/CHiME4_Clean
    spec_type="clean Mel-spectrum"
fi


if [ ${stage} -le 1 ]; then
    echo "stage 1 : preparing testing data"

    for x in $test_list; do
        spec_path=${spec_root}/${x} 
        wave_path=${wav_root}/${x}
        echo "create ${spec_type} from ${wave_path} to ${spec_path}"
        rm -r ${spec_path} 2>/dev/null
        mkdir -p ${spec_path}

        python src/diffwave/preprocess.py ${wave_path} ${spec_path} --${task} --test
    done
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : inference model"
    target_wav_root=${chime4_clean}

    test_spec_list=""
    for x in $test_list; do
        spec_path=${spec_root}/${x}
        test_spec_list="${test_spec_list} ${spec_path}"
    done

    enhanced_path=${diffwave}/Enhanced/${model_name}/model${ckp}/CHiME4
    rm -r ${enhanced_path} 2>/dev/null
    mkdir -p ${enhanced_path} 
    echo "inference enhanced wav file from ${spec_root} to ${enhanced_path}"
    
    python src/diffwave/inference.py  ${diffwave}/${model_name}/weights-${ckp}.pt ${test_spec_list} -o ${enhanced_path}
fi



