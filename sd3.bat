@echo off

:: 設定目錄變數
set TRAIN_DATA_DIR=F:/train/sec
set OUTPUT_DIR=D:/SDXL/SwarmUI/SwarmUI/Models/Lora/REAL
set LOGGING_DIR=D:/SDXL/SwarmUI/SwarmUI/Models/REAL
set OUTPUT_NAME=wang 7
:: 設定步數
set TRAIN_BATCH_SIZE=3
set MAX_TRAIN_STEPS=15000
set /A WARMUP_STEPS=%MAX_TRAIN_STEPS%/10

:: 設定模型變數
set PRETRAINED_MODEL=D:/SDXL/SwarmUI/SwarmUI/Models/Stable-Diffusion/stableDiffusion35_medium.safetensors
set CLIP_L_MODEL=D:/SDXL/SwarmUI/SwarmUI/Models/clip/model.fp16.safetensors
set CLIP_G_MODEL=D:/SDXL/SwarmUI/SwarmUI/Models/clip/clip_g.safetensors
set T5XXL_MODEL=D:/SDXL/SwarmUI/SwarmUI/Models/clip/t5xxl_fp16.safetensors

accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 6 sd3_train_network.py --pretrained_model_name_or_path "%PRETRAINED_MODEL%" --clip_l "%CLIP_L_MODEL%" --clip_g "%CLIP_G_MODEL%" --t5xxl "%T5XXL_MODEL%" --save_model_as safetensors --sdpa  --max_train_steps=%MAX_TRAIN_STEPS% --persistent_data_loader_workers --max_data_loader_n_workers 6 --seed 42 --network_train_unet_only  --gradient_checkpointing --mixed_precision bf16 --save_precision bf16  --network_args  "factor=12" "algo=lokr" "bypass_mode=False" "full_matrix=True" "preset=./preset.toml" --network_module=lycoris.kohya --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base --resolution="512,512" --save_every_n_steps="100"  --train_data_dir "%TRAIN_DATA_DIR%" --output_dir "%OUTPUT_DIR%" --logging_dir "%LOGGING_DIR%"  --output_name "%OUTPUT_NAME%" --loss_type l2 --optimizer_type bitsandbytes.optim.AdEMAMix8bit --optimizer_args "weight_decay=0.1" --lr_scheduler warmup_stable_decay --lr_warmup_steps 0.1 --lr_decay_steps 0.2 --enable_bucket --caption_extension=".txt" --train_batch_size=%TRAIN_BATCH_SIZE%  --cache_latents --cache_latents_to_disk --apply_t5_attn_mask --apply_lg_attn_mask --network_train_unet_only --alpha_mask --max_grad_norm=0.5 --weighting_scheme uniform --training_shift 1 --enable_wildcard --unet_lr=0.0002

tensorboard --logdir="%LOGGING_DIR%"

:: 先按照GUI的標準方式安裝 然後自己修改FLUX.bat的參數
:: 用./venv/Scripts/activate 進入venv再執行FLUX.bat

:::--optimizer_args "warmup_steps=500" "weight_decay=0.1" "use_bias_correction=True" "d_coef=1" "betas=0.9,0.99" "d0=1e-5" "prodigy_steps=1000"  --unet_lr=1 --optimizer_type "prodigyplus.ProdigyPlusScheduleFree"