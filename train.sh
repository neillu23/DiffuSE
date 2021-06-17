export CUDA_VISIBLE_DEVICES='4,7'

stage=2
task="vocoder" #"vocoder" or "se"
model_name="model_vocoder_full"

. ./path.sh

chime4_noisy="${chime4}/isolated/"
chime4_clean="${chime4}/isolated_ext/"
train_list="tr05_bus_simu tr05_caf_simu tr05_ped_simu tr05_str_simu"


if [[ ! " se vocoder " =~ " $task " ]]; then
  echo "Error: \$task must be either se or vocoder: ${task}"
  exit 1;
fi


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
    echo "stage 1 : preparing training data"
    for x in $train_list; do
        spec_path=${spec_root}/${x} 
        wave_path=${wav_root}/${x}

        echo "create ${spec_type} from ${wave_path} to ${spec_path}"

        rm -r ${spec_path} 2>/dev/null
        mkdir -p ${spec_path}

        python src/diffwave/preprocess.py ${wave_path} ${spec_path} --${task}
    done

fi

if [ ${stage} -le 2 ]; then
    echo "stage 2 : training model"
    target_wav_root=${chime4_clean}

    train_spec_list=""
    for x in $train_list; do
        spec_path=${spec_root}/${x}
        train_spec_list="${train_spec_list} ${spec_path}"
    done

    python src/diffwave/__main__.py ${diffwave}/${model_name} ${target_wav_root}  ${train_spec_list}  --${task}
fi

