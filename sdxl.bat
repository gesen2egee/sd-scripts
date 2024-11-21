

accelerate launch --num_cpu_threads_per_process=4 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli2" --resolution="1024,1024" --max_train_steps="15000" --mixed_precision="bf16" --network_args "factor=8" "algo=lokr" "train_norm=True" "bypass_mode=False" "full_matrix=True" "conv_dim=100000" "preset=./preset.toml" --network_dim=100000 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli2" --output_name="good 3" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --caption_dropout_rate=0.1 --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --cache_latents --cache_latents_to_disk --noise_offset=0.0357 --ip_noise_gamma 0.1 --ip_noise_gamma_random_strength --enable_wildcard --train_data_dir="E:/REG" --shuffle_caption --max_data_loader_n_workers 4 --alpha_mask --save_state --save_last_n_steps_state 100 --debiased_estimation_loss  --network_train_unet_only --no_half_vae  --optimizer_args "warmup_steps=500" "weight_decay=0.1" "beta4=0.99" "eps=None" "split_groups=True" "scale_atan2=True" "use_bias_correction=True" "d_coef=1" "betas=0.9,0.99" "d0=1e-5" "prodigy_steps=1000"  --unet_lr=1 --optimizer_type "prodigyplus.ProdigyPlusScheduleFree"

::: --network_weights  " D:/SD/stable-diffusion-webui/models/LyCORIS/loli2/fight 3-step00000750.safetensors"
::: --reg_data_dir="E:/BEST/REAL" --prior_loss_weight=0.7 

:::--network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/loli2/fight 3-step00000750.safetensors" --unet_lr=0.000075 --optimizer_type="AdamWScheduleFree" --optimizer_args "betas=0.9,0.95" "warmup_steps=300" "weight_decay=0.01" 
:::E:/REG  
::: --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_scheduler_min_lr_ratio 0.1 --lr_decay_steps 0.2 --max_grad_norm=0.5
::: --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_scheduler_min_lr_ratio 0.1 --lr_decay_steps 0.2
::: --color_aug --random_crop --prior_loss_weight=0.3 
::: --network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/hypno backup/hypno xl3-step00008000.safetensors"
::: --resume "D:/SD/stable-diffusion-webui/models/LyCORIS/best/best 10-step00000750-state"