export CUDA_VISIBLE_DEVICES='4' 
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix fast none
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix full none
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix fast noisy_out
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix full noisy_out
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix fast noisy_in
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix full noisy_in
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix fast noisy_inout
./inf_vb_noisy.sh 2 291600 se voicebank_model_se_unfix full noisy_inout
./inf_vb_supportive.sh 2 291600 se voicebank_model_se_unfix fast
./inf_vb_supportive.sh 2 291600 se voicebank_model_se_unfix full