export CUDA_VISIBLE_DEVICES='4,7'

stage=2
task="vocoder" #"vocoder" or "se"
model_name="voicebank_model_vocoder"
#pretrain_model="model_vocoder_full/weights-401400.pt"
#fix="--fix2"
. ./path.sh

voicebank_noisy="${voicebank}/noisy_trainset_28spk_wav_16k"
voicebank_clean="${voicebank}/clean_trainset_28spk_wav_16k"


if [[ ! " se vocoder " =~ " $task " ]]; then
  echo "Error: \$task must be either se or vocoder: ${task}"
  exit 1;
fi


if [[ "$task" == "se" ]]; then
    wav_root=${voicebank_noisy}
    spec_root=${diffwave}/spec/voicebank_Noisy
    spec_type="noisy spectrum"

elif [[ "$task" == "vocoder" ]]; then
    wav_root=${voicebank_clean}
    spec_root=${diffwave}/spec/voicebank_Clean
    spec_type="clean Mel-spectrum"
fi

if [ ${stage} -le 1 ]; then
    echo "stage 1 : preparing training data"
    wave_path=${wav_root}
    spec_path=${spec_root} 
    echo "create ${spec_type} from ${wave_path} to ${spec_path}"
    rm -r ${spec_path} 2>/dev/null
    mkdir -p ${spec_path}
    python src/diffwave/preprocess.py ${wave_path} ${spec_path} --${task} --voicebank

fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : training model"
    target_wav_root=${voicebank_clean}

    train_spec_list=""

    spec_path=${spec_root}
    train_spec_list="${train_spec_list} ${spec_path}"
    
    if [ -z "$pretrain_model" ]; then
        python src/diffwave/__main__.py ${diffwave}/${model_name} ${target_wav_root}  ${train_spec_list}  --${task} ${fix}  --voicebank
    else
        python src/diffwave/__main__.py ${diffwave}/${model_name} ${target_wav_root}  ${train_spec_list}  --${task} --pretrain ${diffwave}/${pretrain_model} ${fix}  --voicebank
    fi
fi

