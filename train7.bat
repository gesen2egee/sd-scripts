accelerate launch --num_cpu_threads_per_process=6 "./train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048  --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/ACertainty baked2.safetensors"  --train_data_dir="F:/naruto10/image backup/" --resolution="640,640" --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/sarada"  --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/sarada" --network_alpha="1" --save_model_as=safetensors --network_module=lycoris.kohya --network_args "preset=full" "conv_dim=32" "conv_alpha=1" "module_dropout=0" "use_tucker=False" "use_scalar=True" "rank_dropout_scale=False" "constrain=0.0" "rescaled=True" "algo=diag-oft"  --network_dropout="0" --text_encoder_lr=5e-05 --unet_lr=0.0001 --network_dim=32 --output_name="naruto" --lr_scheduler_num_cycles="1" --debiased_estimation_loss --learning_rate="0.0001" --lr_scheduler="cosine" --train_batch_size="8" --max_train_steps="100000" --mixed_precision="bf16" --save_precision="fp16" --seed="1026" --caption_extension=".txt" --optimizer_type="AdamW" --optimizer_args --lr_scheduler_type "CosineAnnealingWarmRestarts" --lr_scheduler_args "T_0=100" --optimizer_args "weight_decay=0.1" "betas=0.9,0.99" "eps=1e-6" --ip_noise_gamma=0.1 --caption_separator=" " --keep_tokens_separator="|||" --use_object_template --masked_loss --max_grad_norm=5 --keras_aug "rotation_range=40" "width_shift_range=0.2" "height_shift_range=0.2" "shear_range=0.2" "zoom_range=0.2"  --max_grad_norm="0" --max_data_loader_n_workers="6" --clip_skip=2 --caption_dropout_rate="0.1" --bucket_reso_steps=64 --save_every_n_steps="100" --shuffle_caption --gradient_checkpointing --xformers --persistent_data_loader_workers --random_crop --noise_offset=0.0 --debug_dataset