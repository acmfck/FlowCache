export CUDA_VISIBLE_DEVICES=5
model_id="SkyReels-V2/SkyReels-V2-DF-1.3B-540P"

python3 generate_video_df.py \
  --model_id ${model_id} \
  --outdir ./result \
  --expname "flowcache_slow" \
  --resolution 540P \
  --ar_step 5 \
  --causal_block_size 5 \
  --base_num_frames 97 \
  --num_frames 177 \
  --overlap_history 17 \
  --prompt "In a still frame, a weathered stop sign stands prominently at a quiet intersection, its red paint slightly faded and edges rusted, evoking a sense of time passed. The sign is set against a backdrop of a serene suburban street, lined with tall, leafy trees whose branches gently sway in the breeze. The sky above is a soft gradient of twilight hues, transitioning from deep blue to a warm orange, suggesting the end of a peaceful day. The surrounding area is calm, with neatly trimmed lawns and quaint houses, their windows glowing softly with indoor lights, adding to the tranquil atmosphere." \
  --addnoise_condition 20 \
  --inference_steps 50 \
  --seed 1024 \
  --teacache \
  --offload \
  --use_ret_steps \
  --guidance_scale 6 \
  --teacache_thresh 0.1
