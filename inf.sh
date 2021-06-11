export CUDA_VISIBLE_DEVICES='2'
chime4="/home/iis-cvl/DW/CHIME4/"
diffwave="/home/iis-cvl/DW/out"
noisy_list="et05_bus_simu et05_caf_simu et05_ped_simu et05_str_simu et05_bus_real et05_caf_real et05_ped_real et05_str_real"
clean_list="et05_bth"

ckp="89200" 
task="se"

## stage 1: preparing data
#for x in $noisy_list; do
#    rm -r ${diffwave}/CHiME4/${x}_npy || true
#done
#for x in $noisy_list; do
#    echo "create npy under ${diffwave}/CHiME4/${x}_npy"
#    mkdir ${diffwave}/CHiME4/${x}_npy
#    python src/diffwave/preprocess.py ${chime4}/${x} ${diffwave}/CHiME4/${x}_npy --se --test
#done

# stage 2: inference
for x in $noisy_list; do
    rm -r ${diffwave}/Enhanced/${task}/CHiME4/${x} || true
done
for x in $noisy_list; do
    mkdir -p ${diffwave}/Enhanced/${task}/model${ckp}/CHiME4/${x}
    echo "inference enhanced wav file under ${diffwave}/Enhanced/${task}/model${ckp}/CHiME4/${x}"
    python src/diffwave/inference.py  ${diffwave}/model_${task}/weights-${ckp}.pt ${diffwave}/CHiME4/${x}_npy -o ${diffwave}/Enhanced/${task}/model${ckp}/CHiME4/${x}
done



