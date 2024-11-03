
:::accelerate launch --num_cpu_threads_per_process=6 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --max_grad_norm="1" --resolution="1024,1024" --max_train_steps="2000" --mixed_precision="bf16" --network_args  "preset=full" "factor=6" "algo=lokr" "train_norm=True" "bypass_mode=False" --network_dim=100000 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --output_name="sakuranomiya maika lokr" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --train_data_dir="F:/1_FAV/sakuranomiya maika" --unet_lr=0.00005 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1" --network_train_unet_only --caption_dropout_rate=0.1 --caption_tag_dropout_rate=0.05 --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --enable_wildcard --cache_latents --cache_latents_to_disk --timestep_sampling "sigmoid" --ip_noise_gamma 0.03 --no_token_padding --noise_offset=0.06 --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --network_weights "D:/SD/stable-diffusion-webui/models/LyCORIS/best/best lokr 8-step00005750.safetensors"  --discrete_flow_shift 0.5 


accelerate launch --num_cpu_threads_per_process=4 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli" --resolution="1024,1024" --max_train_steps="20000" --mixed_precision="bf16" --network_args  "preset=attn-mlp" "factor=12" "algo=lokr" "train_norm=True" "bypass_mode=False" --network_dim=100000 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/loli" --output_name="real 4" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --unet_lr=0.0002 --optimizer_type="AdamWScheduleFree" --optimizer_args "betas=0.9,0.95" "weight_decay=0.1" --network_train_unet_only --caption_dropout_rate=0.05 --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --cache_latents --cache_latents_to_disk --ip_noise_gamma 0.03 --noise_offset=0.05  --enable_wildcard --train_data_dir="E:/Loli" --shuffle_caption --max_data_loader_n_workers 4 --alpha_mask --max_grad_norm=0.5 --network_weights "D:/SD/stable-diffusion-webui/models/LyCORIS/loli/real 2-step00015000.safetensors" --timestep_sampling "sigmoid"  --debiased_estimation_loss 

:::--debiased_estimation_loss

:::accelerate launch --num_cpu_threads_per_process=4 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --max_grad_norm="1" --resolution="1024,1024" --max_train_steps="3000" --mixed_precision="bf16" --network_args  "preset=full" "factor=8" "algo=lokr" "train_norm=True" "bypass_mode=False" "module_dropout=0.3" --network_dim=100000 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --output_name="azula 4" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --unet_lr=0.0001 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1" --network_train_unet_only --caption_dropout_rate=0.1  --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --cache_latents --cache_latents_to_disk --ip_noise_gamma 0.05 --noise_offset=0.06 --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --timestep_sampling "sigmoid"  --enable_wildcard --train_data_dir="E:/NE"  --shuffle_caption --max_data_loader_n_workers 4   --alpha_mask --network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/best/REG illustration.safetensors" 


::: --network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/best/REG illustration.safetensors"
::: --reg_data_dir="E:/BEST/REAL" --prior_loss_weight=0.7 
:::E:/REG
::: --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_scheduler_min_lr_ratio 0.1 --lr_decay_steps 0.2
::: --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_scheduler_min_lr_ratio 0.1 --lr_decay_steps 0.2
::: --color_aug --random_crop --prior_loss_weight=0.3 
::: --network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/hypno backup/hypno xl3-step00008000.safetensors"
::: --resume "D:/SD/stable-diffusion-webui/models/LyCORIS/best/best 10-step00000750-state"

:::  --network_weights  "D:/SD/stable-diffusion-webui/models/LyCORIS/best/best oft-step00002250.safetensors"
::: accelerate launch --num_cpu_threads_per_process=6 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --max_grad_norm="1" --resolution="1024,1024" --max_train_steps="10000" --mixed_precision="bf16" --network_alpha="1" --network_args  "preset=attn-only" "module_dropout=0.3" "rescaled=True" "algo=diag-oft" "conv_dim=32" "train_norm=True" "constraint=1" --network_dim=32 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --output_name="best" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="500" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --train_data_dir="E:/BEST" --unet_lr=0.00002 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1"  --network_train_unet_only --caption_tag_dropout_rate=0.05 --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --enable_wildcard --cache_latents --cache_latents_to_disk --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --save_state --alpha_mask --ip_noise_gamma 0.1 --ip_noise_gamma_random_strength  

:::accelerate launch --num_cpu_threads_per_process=6 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best new" --max_grad_norm="1" --resolution="1024,1024" --max_train_steps="10000" --mixed_precision="bf16" --network_alpha="16" --network_args "preset=attn-only" "algo=locon" "train_norm=True" --network_dim=32 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best" --output_name="best 14" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --train_data_dir="E:/BEST" --unet_lr=0.0001 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1"  --network_train_unet_only --caption_tag_dropout_rate=0.05 --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --enable_wildcard --cache_latents --cache_latents_to_disk --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --save_state --ip_noise_gamma 0.1 --ip_noise_gamma_random_strength  --debiased_estimation_loss --loss_type huber --huber_schedule snr --huber_c 0.1 --multires_noise_iterations 8 --multires_noise_discount 0.2 --noise_offset 0.02  --resume "D:/SD/stable-diffusion-webui/models/LyCORIS/best/best 12-step00000500-state"


:::accelerate launch --num_cpu_threads_per_process=6 "./sdxl_train_network.py" --bucket_reso_steps=64 --caption_extension=".txt" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=2048 --keep_tokens="5" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/best new" --max_grad_norm="1" --resolution="1024,1024" --max_train_steps="2000" --mixed_precision="bf16" --network_alpha="16" --network_args  "preset=attn-only" "module_dropout=0.3" "rescaled=True" "algo=diag-oft" "conv_dim=32" "train_norm=True" "bypass_mode=False" --network_dim=32 --network_module=lycoris.kohya --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/nohara rin" --output_name="nohara rin 2" --persistent_data_loader_workers --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/illustriousXL_v01.safetensors" --save_every_n_steps="250" --save_model_as=safetensors --save_precision="fp16" --seed="1026" --train_batch_size="3" --train_data_dir="F:/naruto10/aaa/" --unet_lr=0.00005 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1"  --network_train_unet_only --caption_tag_dropout_rate=0.05 --fp8_base --gradient_checkpointing --xformers --save_state_on_train_end --enable_wildcard --cache_latents --cache_latents_to_disk --timestep_sampling "sigmoid" --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --save_state --ip_noise_gamma 0.1 --ip_noise_gamma_random_strength --debiased_estimation_loss
