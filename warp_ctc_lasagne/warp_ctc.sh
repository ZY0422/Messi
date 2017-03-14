THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,lib.cnmem=0.4,warn_float64=warn python train_warp_ctc.py\
 --filename 'warp-ctc'\
 --model 'lstm'\
 --nonlinearitytype 'tanh'\
 --blank_symbol '0'\
 --main_model 'build_warp_ctc_dropout_model'\
 --dropout 'True'\
 --train_scp '/mnt/workspace/xuht/TIMIT/train/train.scp'\
 --feats_scp '/mnt/workspace/xuht/TIMIT/train/feats.scp'\
 --scp_len '/mnt/workspace/xuht/TIMIT/train/len.tmp'\
 --spk2utt '/mnt/workspace/xuht/TIMIT/train/spk2utt'\
 --pkl_data '/mnt/workspace/xuht/TIMIT/train/fbank_cmvn_delta.gzip'\
 --label '/mnt/workspace/xuht/TIMIT/train/labels.cv'\
 --optimizer 'adam'\
 --grad_clip 'clipping'\
 --epoch '500'\
 --mini_batch '10'\
 --rescale '5.0'\
 --momentum '0.9'\
 --model_num '23'\
 --model_id '20170224093101'\
 --n_hidden '250'\
 --fineunte_epoch '60'\
 --grad_type 'warp-ctc'\
 --penalty_type 'None'\
 --param_interput 'None'\
 --eval_dev 'PER'\
 --load_model_flag 'None'\
 --test_model 'train'\
 --ctc_type 'warp-ctc'\
 --data_shuffle 'based-on-lens'\
 --dev_train_scp '/mnt/workspace/xuht/TIMIT/dev/train.scp'\
 --dev_feats_scp '/mnt/workspace/xuht/TIMIT/dev/feats.scp'\
 --dev_scp_len '/mnt/workspace/xuht/TIMIT/dev/len.tmp'\
 --dev_spk2utt '/mnt/workspace/xuht/TIMIT/dev/spk2utt'\
 --dev_pkl_data '/mnt/workspace/xuht/TIMIT/dev/fbank_cmvn_delta.gzip'\
 --dev_label '/mnt/workspace/xuht/TIMIT/dev/labels.cv'\
 --phoneme2int '/mnt/workspace/xuht/TIMIT/lang_phn/units.txt'\
 --phoneme_maps '/mnt/workspace/xuht/TIMIT/phones.60-48-39.map'\
 --output_phoneme '/mnt/workspace/xuht/TIMIT/output_phoneme.txt'\
 --transform_phoneme '/mnt/workspace/xuht/TIMIT/output_phoneme.txt'