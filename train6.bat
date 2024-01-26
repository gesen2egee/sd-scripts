accelerate launch --num_cpu_threads_per_process=6 "./train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/model-001.safetensors" --train_data_dir="F:/one piece9/image" --reg_data_dir="f:/good/" --resolution="640,640" --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/naru" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/naru" --network_module=lycoris.kohya --network_alpha="1" --save_model_as=safetensors --network_args "preset=full" "algo=diag-oft" "rescaleda=True" "conv_dim=32" "train_norm=Ture" "constrain=5e-5" --caption_separator " " --keep_tokens_separator "|||"  --text_encoder_lr=0.000015 --unet_lr=0.0001 --network_dim=64 --output_name="naru emb1" --learning_rate="0.000015" --lr_warmup_steps="300" --train_batch_size="8" --max_train_steps="1" --mixed_precision="bf16" --save_precision="fp16" --seed="1026" --caption_extension=".txt" --cache_latents --cache_latents_to_disk --optimizer_type="Adamw8bit" --caption_dropout_rate="0.1" --max_data_loader_n_workers="6" --clip_skip=2 --keep_tokens="8" --bucket_reso_steps=64 --save_every_n_steps="100" --shuffle_caption --gradient_checkpointing --xformers --persistent_data_loader_workers  --caption_separator " " --ip_noise_gamma=0.1 --enable_ema --ema_decay=0.9999 --lr_scheduler "REX" --debiased_estimation_loss
 
accelerate launch --num_cpu_threads_per_process=6 "./train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 --pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/model-001.safetensors" --train_data_dir="F:/myhero/image" --reg_data_dir="f:/good/" --resolution="640,640" --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/naru" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/naru" --network_module=lycoris.kohya --network_alpha="1" --save_model_as=safetensors --network_args "preset=full" "algo=diag-oft" "rescaleda=True" "conv_dim=32" "train_norm=Ture" "constrain=5e-5" --caption_separator " " --keep_tokens_separator "|||"  --text_encoder_lr=0.000015 --unet_lr=0.0001 --network_dim=64 --output_name="naru emb1" --learning_rate="0.000015" --lr_warmup_steps="300" --train_batch_size="8" --max_train_steps="1" --mixed_precision="bf16" --save_precision="fp16" --seed="1026" --caption_extension=".txt" --cache_latents --cache_latents_to_disk --optimizer_type="Adamw8bit" --caption_dropout_rate="0.1" --max_data_loader_n_workers="6" --clip_skip=2 --keep_tokens="8" --bucket_reso_steps=64 --save_every_n_steps="100" --shuffle_caption --gradient_checkpointing --xformers --persistent_data_loader_workers  --caption_separator " " --ip_noise_gamma=0.1 --enable_ema --ema_decay=0.9999 --lr_scheduler "REX" --debiased_estimation_loss

accelerate launch --num_cpu_threads_per_process=6 "./train_network.py" --enable_bucket --min_bucket_reso=256 --max_bucket_reso=2048 
--pretrained_model_name_or_path="D:/SD/stable-diffusion-webui/models/Stable-diffusion/model-001.safetensors" --train_data_dir="F:/dragonball/image" --reg_data_dir="f:/good/" --resolution="640,640" --output_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/naru" --logging_dir="D:/SD/stable-diffusion-webui/models/LyCORIS/naru" --network_module=lycoris.kohya --network_alpha="1" --save_model_as=safetensors --network_args "preset=full" "algo=diag-oft" "rescaleda=True" "conv_dim=32" "train_norm=Ture" "constrain=5e-5" --caption_separator " " --keep_tokens_separator "|||"  --text_encoder_lr=0.000015 --unet_lr=0.0001 --network_dim=64 --output_name="naru emb1" --learning_rate="0.000015" --lr_warmup_steps="300" --train_batch_size="8" --max_train_steps="1" --mixed_precision="bf16" --save_precision="fp16" --seed="1026" --caption_extension=".txt" --cache_latents --cache_latents_to_disk --optimizer_type="Adamw8bit" --caption_dropout_rate="0.1" --max_data_loader_n_workers="6" --clip_skip=2 --keep_tokens="8" --bucket_reso_steps=64 --save_every_n_steps="100" --shuffle_caption --gradient_checkpointing --xformers --persistent_data_loader_workers  --caption_separator " " --ip_noise_gamma=0.1 --enable_ema --ema_decay=0.9999 --lr_scheduler "REX" --debiased_estimation_loss 