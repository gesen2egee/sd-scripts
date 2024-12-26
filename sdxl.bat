
accelerate launch --num_cpu_threads_per_process=4 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli4" --resolution="1024,1024" --max_train_steps="2000" --mixed_precision="bf16"  --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli3" --output_name="sakuranomiya maika 036" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="100" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --cache_latents --cache_latents_to_disk --debiased_estimation_loss  --ip_noise_gamma 0.03  --noise_offset 0.06 --train_data_dir="F:/1_FAV"  --max_data_loader_n_workers 4 --save_state --save_last_n_steps_state 100 --no_half_vae --network_args "factor=8" "algo=lokr" "bypass_mode=False" "use_scalar=Ture" "full_matrix=True" "conv_dim=100000" "preset=full" --network_dim 100000 --caption_dropout_rate=0.05 --clip_skip 2 --unet_lr=0.001 --max_grad_norm=1 --optimizer_type SGD_sai --network_train_unet_only --lr_scheduler_type "custom.custom.RexWithRestarts"  --enable_wildcard --shuffle_caption



:::--unet_lr=0.0005 --max_grad_norm=1 --optimizer_type adamw8bit --optimizer_args "weight_decay=0.1" --network_train_unet_only --lr_scheduler_type "custom.custom.RexWithRestarts"  --enable_wildcard --shuffle_caption


:::accelerate launch --num_cpu_threads_per_process=4 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli4" --resolution="1024,1024" --max_train_steps="100000" --mixed_precision="bf16"  --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli3" --output_name="real 008" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --cache_latents --cache_latents_to_disk --ip_noise_gamma 0.03  --noise_offset 0.06 --train_data_dir="e:/real"  --max_data_loader_n_workers 4 --save_state --save_last_n_steps_state 100 --debiased_estimation_loss  --no_half_vae --network_args "factor=8" "algo=lokr" "train_norm=True" "bypass_mode=False" "use_scalar=Ture" "full_matrix=True" "conv_dim=100000" "preset=preset.toml" --caption_dropout_rate=0.05 --clip_skip 2  --unet_lr=0.0001 --max_grad_norm=0 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1" --network_train_unet_only --lr_scheduler_type "custom.custom.RexWithRestarts" --lr_scheduler_args "first_cycle=1000"  --enable_wildcard --shuffle_caption



::: --network_args  "preset=attn-only" "rescaled=True" "algo=diag-oft" "train_norm=True" "bypass_mode=False" "use_scalar=Ture" "constraint=1e-3" --network_dim=32--enable_wildcard --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors"


:::--clip_skip 2 
:::  --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --lr_scheduler_min_lr_ratio 0.1  --alpha_mask 


::: --optimizer_args "warmup_steps=500" "prodigy_steps=1250" "weight_decay=0.1" "use_bias_correction=True" "factored=True" "d_coef=1" "betas=0.9,0.99" "d0=1e-5" --unet_lr=1 --optimizer_type "prodigyplus.ProdigyPlusScheduleFree"
::: --lr_scheduler_min_lr_ratio 0.1 -alpha_mask --enable_wildcard --reg_data_dir="E:/newreg" ./preset.toml
::: --network_weights  " D:/SD/stable-diffusion-webui/models/LyCORIS/loli2/fight 3-step00000750.safetensors"
::: --reg_data_dir="E:/BEST/REAL" --prior_loss_weight=0.7 --enable_wildcard

:::--network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/loli2/fight 3-step00000750.safetensors" --unet_lr=0.00005 --optimizer_type="AdamWScheduleFree" --optimizer_args "betas=0.9,0.95" "warmup_steps=300" "weight_decay=0.01" 
:::E:/REG  
::: --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_scheduler_min_lr_ratio 0.1 --lr_decay_steps 0.2 
::: --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_scheduler_min_lr_ratio 0.1 --lr_decay_steps 0.2
::: --color_aug --random_crop --prior_loss_weight=0.3 
::: --network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/hypno backup/hypno xl3-step00008000.safetensors"
::: --resume "D:/SD/stable-diffusion-webui/models/LyCORIS/best/best 10-step00000750-state"