accelerate launch --num_cpu_threads_per_process=6 "./train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned.safetensors" --train_data_dir="F:/train/sec" --reg_data_dir="F:/train/Regreal" --resolution="768,768" --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/secsec" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/secsec"  --network_alpha="1" --save_model_as=safetensors --network_module=lycoris.kohya --network_args "preset=full" "algo=diag-oft" "rescaled=True" "train_norm=Ture" "constrain=5e-5" "conv_dim=32" "conv_alpha=1" "rank_dropout_scale=true" "rank_dropout=0.3" "module_dropout=0.3"  --text_encoder_lr=1.0 --unet_lr=1.0 --network_dim=32 --output_name="secsec Prodigyzz1" --min_snr_gamma=5 --learning_rate="1.0" --lr_scheduler="cosine" --lr_warmup_steps="600" --train_batch_size="8" --max_train_steps="6000" --mixed_precision="bf16" --save_precision="fp16" --seed="1026" --caption_extension=".txt" --cache_latents --cache_latents_to_disk --optimizer_type="Prodigy" --optimizer_args "safeguard_warmup=True" "decouple=True"  "use_bias_correction=True" "betas=0.9,0.99" --caption_dropout_rate="0.1" --max_data_loader_n_workers="6" --clip_skip=2 --keep_tokens="8" --bucket_reso_steps=64 --save_every_n_steps="100" --shuffle_caption --gradient_checkpointing --xformers --persistent_data_loader_workers  --caption_separator " " --ip_noise_gamma=0.1  --enable_ema --ema_decay=0.9995 --embeddings="F:/naruto7/test/uzumaki-himawari.pt" --continue_inversion --embedding_lr=1.0 --enable_ema --ema_decay=0.9995  --lr_scheduler "REX"