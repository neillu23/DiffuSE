export CUDA_VISIBLE_DEVICES='1' 

stage=2
ckp="248712"
task="vocoder" #"vocoder" or "se"
model_name="voicebank_model_vocoder"

. ./path.sh

voicebank_noisy="${voicebank}/noisy_testset_wav_16k"
voicebank_clean="${voicebank}/clean_testset_wav_16k"


if [[ ! " se vocoder " =~ " $task " ]]; then
  echo "Error: \$task must be either se or vocoder: ${task}"
  exit 1;
fi


if [[ "$task" == "se" ]]; then
    wav_root=${voicebank_noisy}
    spec_root=${diffwave}/spec/voicebank_Noisy_Test
    spec_type="noisy spectrum"

elif [[ "$task" == "vocoder" ]]; then
    wav_root=${voicebank_clean}
    spec_root=${diffwave}/spec/voicebank_Clean_Test
    spec_type="clean Mel-spectrum"
fi


if [ ${stage} -le 1 ]; then
    echo "stage 1 : preparing testing data"

    spec_path=${spec_root}
    wave_path=${wav_root}
    echo "create ${spec_type} from ${wave_path} to ${spec_path}"
    rm -r ${spec_path} 2>/dev/null
    mkdir -p ${spec_path}
    python src/diffwave/preprocess.py ${wave_path} ${spec_path} --${task} --test --voicebank
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : inference model"
    target_wav_root=${voicebank_clean}

    test_spec_list=${spec_root}
    
    enhanced_path=${diffwave}/Enhanced/${model_name}/model${ckp}/
    rm -r ${enhanced_path} 2>/dev/null
    mkdir -p ${enhanced_path} 
    echo "inference enhanced wav file from ${spec_root} to ${enhanced_path}"
    
    python src/diffwave/inference.py  ${diffwave}/${model_name}/weights-${ckp}.pt ${test_spec_list} -o ${enhanced_path} --voicebank
fi