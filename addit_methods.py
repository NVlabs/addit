# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import gc
import torch
from visualization_utils import show_images

def _add_object(
    pipe,
    prompts,
    seed_src,
    seed_obj,
    extended_scale,
    source_latents,
    structure_transfer_step,
    subject_token,
    blend_steps,
    show_attention=False,
    localization_model="attention_points_sam",
    is_img_src=False,
    img_src_latents=None,
    use_offset=False,
    display_output=False,
):
    gc.collect()
    torch.cuda.empty_cache()

    out = pipe( 
        prompt=prompts,
        guidance_scale=3.5 if (not is_img_src) else [1,3.5],
        height=1024,
        width=1024,
        max_sequence_length=512,
        num_inference_steps=30,
        seed=[seed_src, seed_obj],
        
        # Extended Attention
        extended_scale=extended_scale,
        extended_steps_multi=10,
        extended_steps_single=20,
        
        # Structure Transfer
        source_latents=source_latents,
        structure_transfer_step=structure_transfer_step,
        
        # Latent Blending
        subject_token=subject_token,
        localization_model=localization_model,
        blend_steps=blend_steps,
        show_attention=show_attention,
        
        # Real Image Source
        is_img_src=is_img_src,
        img_src_latents=img_src_latents,
        use_offset=use_offset,
    )

    if display_output:
        show_images(out.images)

    return out.images

def add_object_generated(
    pipe,
    prompt_source,
    prompt_object,
    subject_token,
    seed_src,
    seed_obj,
    show_attention=False,
    extended_scale=1.05,
    structure_transfer_step=2,
    blend_steps=[15],
    localization_model="attention_points_sam",
    display_output=False
):
    gc.collect()
    torch.cuda.empty_cache()

    # Generate source image and latents for each seed1
    print('Generating source image...')
    source_image, source_latents = pipe(
        prompt=[prompt_source],
        guidance_scale=3.5,
        height=1024,
        width=1024,
        max_sequence_length=512,
        num_inference_steps=30,
        seed=[seed_src],
        output_type="both",
    )
    source_image = source_image[0]

    # Run the core combination logic
    print('Running Addit...')
    src_image, edited_image = _add_object(
        pipe=pipe,
        prompts=[prompt_source, prompt_object],
        subject_token=subject_token,
        seed_src=seed_src,
        seed_obj=seed_obj,
        source_latents=source_latents,
        structure_transfer_step=structure_transfer_step,
        extended_scale=extended_scale,
        blend_steps=blend_steps,
        show_attention=show_attention,
        localization_model=localization_model,
        display_output=display_output
    )

    return src_image, edited_image

def add_object_real(
    pipe,
    source_image,
    prompt_source,
    prompt_object,
    subject_token,
    seed_src,
    seed_obj,
    localization_model="attention_points_sam",
    extended_scale=1.05,
    structure_transfer_step=4,
    blend_steps=[20],
    use_offset=False,
    show_attention=False,
    use_inversion=False,
    display_output=False
):
    print('Noising-Denoising Original Image')
    gc.collect()
    torch.cuda.empty_cache()

    # Get initial latents
    source_latents = pipe.call_img2img(
        prompt=prompt_source,
        image=source_image,
        num_inference_steps=30,
        strength=0.1,
        guidance_scale=3.5,
        output_type="latent",
        generator=torch.Generator(device=pipe.device).manual_seed(0)
    ).images

    # Optional inversion step
    img_src_latents = None
    if use_inversion:
        print('Inverting Image')
        gc.collect()
        torch.cuda.empty_cache()

        latents_list = pipe.call_invert(
            prompt=prompt_source,
            image=source_latents,
            num_inference_steps=30,
            guidance_scale=1,
            fixed_point_iterations=2,
            generator=torch.Generator(device=pipe.device).manual_seed(0)
        )
        img_src_latents = [x[0] for x in latents_list][::-1]

    print('Running Addit')
    gc.collect()
    torch.cuda.empty_cache()

    src_image, edited_image = _add_object(
        pipe,
        prompts=[prompt_source, prompt_object],
        seed_src=seed_src,
        seed_obj=seed_obj,
        extended_scale=extended_scale,
        source_latents=source_latents,
        structure_transfer_step=structure_transfer_step,
        subject_token=subject_token,
        blend_steps=blend_steps,
        show_attention=show_attention,
        localization_model=localization_model,
        is_img_src=True,
        img_src_latents=img_src_latents,
        use_offset=use_offset,
        display_output=display_output,
    )

    return src_image, edited_image
