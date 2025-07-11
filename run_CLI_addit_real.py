#!/usr/bin/env python3
# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import os
import argparse
import torch
import random
from PIL import Image

from visualization_utils import show_images
from addit_flux_pipeline import AdditFluxPipeline
from addit_flux_transformer import AdditFluxTransformer2DModel
from addit_scheduler import AdditFlowMatchEulerDiscreteScheduler
from addit_methods import add_object_real

def main():
    parser = argparse.ArgumentParser(description='Run ADDIT with real images')
    
    # Required arguments
    parser.add_argument('--source_image', type=str, default="images/bed_dark_room.jpg",
                        help='Path to the source image')
    parser.add_argument('--prompt_source', type=str, default="A photo of a bed in a dark room",
                        help='Source prompt describing the original image')
    parser.add_argument('--prompt_target', type=str, default="A photo of a dog lying on a bed in a dark room",
                        help='Target prompt describing the desired edited image')
    parser.add_argument('--subject_token', type=str, default="dog",
                        help='Subject token to add to the image')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save output images (default: outputs)')
    parser.add_argument('--seed_src', type=int, default=6311,
                        help='Seed for source generation')
    parser.add_argument('--seed_obj', type=int, default=1,
                        help='Seed for edited image generation')
    parser.add_argument('--extended_scale', type=float, default=1.1,
                        help='Extended attention scale (default: 1.1)')
    parser.add_argument('--structure_transfer_step', type=int, default=4,
                        help='Structure transfer step (default: 4)')
    parser.add_argument('--blend_steps', type=int, nargs='*', default=[18],
                        help='Blend steps (default: [18])')
    parser.add_argument('--localization_model', type=str, default="attention",
                        help='Localization model (default: attention, Options: [attention_points_sam, attention, attention_box_sam, attention_mask_sam, grounding_sam])')
    parser.add_argument('--use_offset', action='store_true',
                        help='Use offset in processing')
    parser.add_argument('--show_attention', action='store_true',
                        help='Show attention maps')
    parser.add_argument('--disable_inversion', action='store_true',
                        help='Disable source image inversion')
    parser.add_argument('--display_output', action='store_true',
                        help='Display output images during processing')
    
    args = parser.parse_args()
    
    assert args.subject_token in args.prompt_target, "Subject token must appear in the prompt_target"

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    my_transformer = AdditFluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    )
    
    pipe = AdditFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        transformer=my_transformer,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    pipe.scheduler = AdditFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Load and resize source image
    source_image = Image.open(args.source_image).resize((1024, 1024))
    print(f"Loaded source image: {args.source_image}")
    
    # Set random seed
    if args.seed_src is None:
        random.seed(0)
        args.seed_src = random.randint(0, 10000)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the seeds
    print(f"\nProcessing with source seed: {args.seed_src}, object seed: {args.seed_obj}")
    
    src_image, edited_image = add_object_real(
        pipe, 
        source_image=source_image, 
        prompt_source=args.prompt_source, 
        prompt_object=args.prompt_target, 
        subject_token=args.subject_token, 
        seed_src=args.seed_src, 
        seed_obj=args.seed_obj, 
        extended_scale=args.extended_scale, 
        structure_transfer_step=args.structure_transfer_step, 
        blend_steps=args.blend_steps,
        localization_model=args.localization_model,
        use_offset=args.use_offset, 
        show_attention=args.show_attention, 
        use_inversion=not args.disable_inversion, 
        display_output=args.display_output
    )
    
    # Save output images
    src_filename = f"src_{args.prompt_source}_seed-src={args.seed_src}.png"
    edited_filename = f"edited_{args.prompt_target}_seed-src={args.seed_src}_seed-obj={args.seed_obj}.png"
        
    src_image.save(os.path.join(args.output_dir, src_filename))
    edited_image.save(os.path.join(args.output_dir, edited_filename))
    
    print(f"Saved images: {src_filename}, {edited_filename}")

if __name__ == "__main__":
    main() 