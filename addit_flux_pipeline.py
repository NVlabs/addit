# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

from typing import Any, Callable, Dict, List, Optional, Union
import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
import matplotlib.pyplot as plt

import torch.fft
import torch.nn.functional as F

from diffusers.models.attention_processor import FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
from addit_attention_processors import AdditFluxAttnProcessor2_0, AdditFluxSingleAttnProcessor2_0
from addit_attention_store import AttentionStore

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from skimage import filters
from visualization_utils import show_image_and_heatmap, show_images, draw_points_on_pil_image, draw_bboxes_on_image
from addit_blending_utils import clipseg_predict, grounding_sam_predict, mask_to_box_sam_predict, \
            mask_to_mask_sam_predict, attention_to_points_sam_predict

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scipy.optimize import brentq
from scipy.optimize import root_scalar

def register_my_attention_processors(transformer, attention_store, extended_steps_multi, extended_steps_single):
    attn_procs = {}
    
    for i, (name, processor) in enumerate(transformer.attn_processors.items()):
        layer_name = ".".join(name.split(".")[:2])

        if layer_name.startswith("transformer_blocks"):
            attn_procs[name] = AdditFluxAttnProcessor2_0(layer_name=layer_name, 
                                                      attention_store=attention_store, 
                                                      extended_steps=extended_steps_multi)
        elif layer_name.startswith("single_transformer_blocks"):
            attn_procs[name] = AdditFluxSingleAttnProcessor2_0(layer_name=layer_name, 
                                                            attention_store=attention_store, 
                                                            extended_steps=extended_steps_single)

    transformer.set_attn_processor(attn_procs)

def register_regular_attention_processors(transformer):
    attn_procs = {}

    for i, (name, processor) in enumerate(transformer.attn_processors.items()):
        layer_name = ".".join(name.split(".")[:2])

        if layer_name.startswith("transformer_blocks"):
            attn_procs[name] = FluxAttnProcessor2_0()
        elif layer_name.startswith("single_transformer_blocks"):
            attn_procs[name] = FluxSingleAttnProcessor2_0()

    transformer.set_attn_processor(attn_procs)

def img2img_retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class AdditFluxPipeline(FluxPipeline):
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            latents = torch.empty(shape, device=device, dtype=dtype)

            latents_list = [randn_tensor(shape, generator=g, device=device, dtype=dtype) for g in generator]
            
            for i, l_i in enumerate(latents_list):
                latents[i] = l_i[i]
        else:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        return latents, latent_image_ids
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,

        seed: Optional[Union[int, List[int]]] = None,
        same_latent_for_all_prompts: bool = False,

        # Extended Attention
        extended_steps_multi: Optional[int] = -1,
        extended_steps_single: Optional[int] = -1,
        extended_scale: Optional[Union[float, str]] = 1.0,

        # Structure Transfer
        source_latents: Optional[torch.FloatTensor] = None,
        structure_transfer_step: int = 5,

        # Latent Blending
        subject_token: Optional[str] = None,
        localization_model: Optional[str] = "attention_points_sam",
        blend_steps: List[int] = [],
        show_attention: bool = False,

        # Real Image Source
        is_img_src: bool = False,
        use_offset: bool = False,
        img_src_latents: Optional[List[torch.FloatTensor]] = None,        
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        device = self._execution_device

        # Blend Steps
        blend_models = {}
        if len(blend_steps) > 0:
            if localization_model == "clipseg":
                blend_models["clipseg_processor"] = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
                blend_models["clipseg_model"] = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
            elif localization_model == "grounding_sam":
                grounding_dino_model_id = "IDEA-Research/grounding-dino-base"
                blend_models["grounding_processor"] = AutoProcessor.from_pretrained(grounding_dino_model_id)
                blend_models["grounding_model"] = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(device)
                blend_models["sam_predictor"] = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            elif localization_model == "clipseg_sam":
                blend_models["clipseg_processor"] = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
                blend_models["clipseg_model"] = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
                blend_models["sam_predictor"] = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
            elif localization_model == "attention":
                pass
            elif localization_model in ["attention_box_sam", "attention_mask_sam", "attention_points_sam"]:
                blend_models["sam_predictor"] = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        if (generator is None) and seed is not None:
            if isinstance(seed, int):
                generator = torch.Generator(device=device).manual_seed(seed)
            else:
                assert len(seed) == batch_size, "The number of seeds must match the batch size"
                generator = [torch.Generator(device=device).manual_seed(s) for s in seed]

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )             
        
        if same_latent_for_all_prompts:
            latents = latents[:1].repeat(batch_size * num_images_per_prompt, 1, 1)

        noise = latents.clone()
        
        attention_store_kwargs = {}

        if extended_scale == "auto":
            is_auto_extend_scale = True
            extended_scale = 1.05
            attention_store_kwargs["is_cache_attn_ratio"] = True
            auto_extended_step = 5
            target_auto_ratio = 1.05
        else:
            is_auto_extend_scale = False

        if len(blend_steps) > 0:
            attn_steps = range(blend_steps[0] - 2, blend_steps[0] + 1)
            attention_store_kwargs["record_attention_steps"] = attn_steps

        self.attention_store = AttentionStore(prompts=prompt, tokenizer=self.tokenizer_2, subject_token=subject_token, **attention_store_kwargs)
        register_my_attention_processors(self.transformer, self.attention_store, extended_steps_multi, extended_steps_single)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            if isinstance(guidance_scale, float):
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            elif isinstance(guidance_scale, list):
                assert len(guidance_scale) == latents.shape[0], "The number of guidance scales must match the batch size"
                guidance = torch.tensor(guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        if is_img_src and img_src_latents is None:
            assert source_latents is not None, "source_latents must be provided when is_img_src is True"

            rand_noise = noise[0].clone()
            img_src_latents = []

            for i in range(timesteps.shape[0]):
                sigma = self.scheduler.sigmas[i]
                img_src_latents.append((1.0 - sigma) * source_latents[0] + sigma * rand_noise)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # For denoising from source image
                if is_img_src:
                    latents[0] = img_src_latents[i]

                # For Structure Transfer
                if (source_latents is not None) and i == structure_transfer_step:
                    sigma = self.scheduler.sigmas[i]
                    latents[1] = (1.0 - sigma) * source_latents[0] + sigma * noise[1]                  

                if is_auto_extend_scale and i == auto_extended_step:
                    def f(gamma):
                        self.attention_store.attention_ratios[i] = {}
                        noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            proccesor_kwargs={"step_index": i, "extended_scale": gamma},
                        )[0]

                        scores_per_layer = self.attention_store.get_attention_ratios(step_indices=[i], display_imgs=False)
                        source_sum, text_sum, target_sum = scores_per_layer['transformer_blocks']

                        # We want to find the gamma that makes the ratio equal to K
                        ratio = (target_sum / source_sum)
                        return (ratio - target_auto_ratio)
                    
                    gamma_sol = brentq(f, 1.0, 1.2, xtol=0.01)

                    print('Chosen gamma:', gamma_sol)
                    extended_scale = gamma_sol              
                else:
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        proccesor_kwargs={"step_index": i, "extended_scale": extended_scale},
                    )[0]   

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents, x0 = self.scheduler.step(noise_pred, t, latents, return_dict=False, step_index=i)

                if use_offset and is_img_src and (i+1 < len(img_src_latents)):
                    next_latent = img_src_latents[i+1]
                    offset = (next_latent - latents[0])
                    latents[1] = latents[1] + offset

                # blend latents
                if i in blend_steps and (subject_token is not None) and (localization_model is not None):
                    x0 = self._unpack_latents(x0, height, width, self.vae_scale_factor)
                    x0 = (x0 / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    images = self.vae.decode(x0, return_dict=False)[0]
                    images = self.image_processor.postprocess(images, output_type="pil")

                    self.do_step_blend(images, latents, subject_token, localization_model, show_attention, i, blend_models)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if output_type == "latent":
            image = latents
        elif output_type == "both":
            return_latents = latents
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)

            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type="pil")
            
            return (image, return_latents)
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)

            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    def do_step_blend(self, images, latents, subject_token, localization_model, 
                      show_attention, i, blend_models):
        
        device = latents.device
        latents_dtype = latents.dtype

        clipseg_processor = blend_models.get("clipseg_processor", None)
        clipseg_model = blend_models.get("clipseg_model", None)
        grounding_processor = blend_models.get("grounding_processor", None)
        grounding_model = blend_models.get("grounding_model", None)
        sam_predictor = blend_models.get("sam_predictor", None)

        image_to_display = []
        titles_to_display = []

        if show_attention:
            image_to_display += [images[0], images[1]]
            titles_to_display += ["Source X0", "Target X0"]

        if localization_model == "clipseg":
            subject_mask = clipseg_predict(clipseg_model, clipseg_processor, [images[-1]], f"A photo of {subject_token}", device)
        elif localization_model == "grounding_sam":
            subject_mask = grounding_sam_predict(grounding_model, grounding_processor, sam_predictor, images[-1], f"A {subject_token}.", device)
        elif localization_model == "clipseg_sam":
            subject_mask = clipseg_predict(clipseg_model, clipseg_processor, [images[-1]], f"A photo of {subject_token}", device)
            subject_mask = mask_to_box_sam_predict(subject_mask, sam_predictor, images[-1], None, device)             
        elif localization_model == "attention":
            store = self.attention_store.image2text_store
            attention_maps, attention_masks, tokens = self.attention_store.aggregate_attention(store, target_layers=None, gaussian_kernel=3)

            subject_mask = attention_masks[0][-1].to(device)
            subject_attention = attention_maps[0][-1].to(device)

            if show_attention:
                attentioned_image = show_image_and_heatmap(subject_attention.float(), images[1], relevnace_res=512)
                attention_masked_image = show_image_and_heatmap(subject_mask.float(), images[1], relevnace_res=512)
                
                image_to_display += [attentioned_image, attention_masked_image]
                titles_to_display += ["Attention", "Attention Mask"]
            
        elif localization_model == "attention_box_sam":
            store = self.attention_store.image2text_store
            attention_maps, attention_masks, tokens = self.attention_store.aggregate_attention(store, target_layers=None, gaussian_kernel=3)

            attention_mask = attention_masks[0][-1].to(device)
            subject_attention = attention_maps[0][-1].to(device)

            subject_mask, bbox = mask_to_box_sam_predict(attention_mask, sam_predictor, images[-1], None, device)

            if show_attention:
                attentioned_image = show_image_and_heatmap(subject_attention.float(), images[1], relevnace_res=512)
                attention_masked_image = show_image_and_heatmap(attention_mask.float(), images[1], relevnace_res=512)

                sam_masked_image = show_image_and_heatmap(subject_mask.float(), images[1], relevnace_res=1024)
                sam_masked_image = draw_bboxes_on_image(sam_masked_image, [bbox.tolist()], color="green", thickness=5)

                image_to_display += [attentioned_image, attention_masked_image, sam_masked_image]
                titles_to_display += ["Attention", "Attention Mask", "SAM Mask"]

        elif localization_model == "attention_mask_sam":
            store = self.attention_store.image2text_store
            attention_maps, attention_masks, tokens = self.attention_store.aggregate_attention(store, target_layers=None, gaussian_kernel=3)

            attention_mask = attention_masks[0][-1].to(device)
            subject_attention = attention_maps[0][-1].to(device)

            subject_mask = mask_to_mask_sam_predict(attention_mask, sam_predictor, images[-1], None, device)

            if show_attention:
                print('Attention:')
                attentioned_image = show_image_and_heatmap(subject_attention.float(), images[1], relevnace_res=512)
                attention_masked_image = show_image_and_heatmap(attention_mask.float(), images[1], relevnace_res=512)
                sam_masked_image = show_image_and_heatmap(subject_mask.float(), images[1], relevnace_res=1024)

                image_to_display += [attentioned_image, attention_masked_image, sam_masked_image]
                titles_to_display += ["Attention", "Attention Mask", "SAM Mask"]

        elif localization_model == "attention_points_sam":
            store = self.attention_store.image2text_store
            attention_maps, attention_masks, tokens = self.attention_store.aggregate_attention(store, target_layers=None, gaussian_kernel=3)

            attention_mask = attention_masks[0][-1].to(device)
            subject_attention = attention_maps[0][-1].to(device)

            subject_mask, point_coords = attention_to_points_sam_predict(subject_attention, attention_mask, sam_predictor, images[1], None, device)

            if show_attention:
                print('Attention:')
                attentioned_image = show_image_and_heatmap(subject_attention.float(), images[1], relevnace_res=512)
                attention_masked_image = show_image_and_heatmap(attention_mask.float(), images[1], relevnace_res=512)

                sam_masked_image = show_image_and_heatmap(subject_mask.float(), images[1], relevnace_res=1024)
                sam_masked_image = draw_points_on_pil_image(sam_masked_image, point_coords, point_color="green", radius=10)

                image_to_display += [attentioned_image, attention_masked_image, sam_masked_image]
                titles_to_display += ["Attention", "Attention Mask", "SAM Mask"]

        if show_attention:
            show_images(image_to_display, titles_to_display, size=512, save_path="attn_vis.png")

        # Resize the mask to latents size
        latents_mask = torch.nn.functional.interpolate(subject_mask.view(1,1,subject_mask.shape[-2],subject_mask.shape[-1]), size=64, mode='bilinear').view(4096, 1).to(latents_dtype)
        latents_mask[latents_mask > 0.01] = 1
                    
        latents[1] = latents[1] * latents_mask + latents[0] * (1 - latents_mask)

    ############# Image to Image Methods #############
    def img2img_encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                img2img_retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = img2img_retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents
    
    def img2img_prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self.img2img_prepare_latent_image_ids(batch_size, height, width, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self.img2img_encode_vae_image(image=image, generator=generator)
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids
    
    def img2img_check_inputs(
        self,
        prompt,
        prompt_2,
        strength,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def img2img_get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start
    
    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def img2img_prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
        
    @torch.no_grad()
    def call_img2img(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.img2img_check_inputs(
            prompt,
            prompt_2,
            strength,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Preprocess image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        register_regular_attention_processors(self.transformer)

        # 4.Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.img2img_get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        latents, latent_image_ids = self.img2img_prepare_latents(
            init_image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        text_ids = text_ids.expand(latents.shape[0], -1, -1)
        latent_image_ids = latent_image_ids.expand(latents.shape[0], -1, -1)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

    ############# Invert Methods #############
    def invert_prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        add_noise=False,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self.img2img_encode_vae_image(image=image, generator=generator)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)
        
        if add_noise:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            latents = image_latents

        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        
        return latents, latent_image_ids
    
    @torch.no_grad()
    def call_invert(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,

        fixed_point_iterations: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 1.5. Preprocess image
        if isinstance(image, Image.Image):
            init_image = self.image_processor.preprocess(image, height=height, width=width)
        elif isinstance(image, torch.Tensor):
            init_image = image
            latents = image
        else:
            raise ValueError("Image must be of type `PIL.Image.Image` or `torch.Tensor`")

        init_image = init_image.to(dtype=torch.float32)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        # latents, latent_image_ids = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        latents, latent_image_ids = self.invert_prepare_latents(
            init_image,
            None,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            False
        )

        register_regular_attention_processors(self.transformer)
        
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )

        # For Inversion, reverse the sigmas
        # sigmas = sigmas[::-1]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        self.scheduler.sigmas = reversed(self.scheduler.sigmas)

        timesteps_zero_start = reversed(torch.cat([self.scheduler.timesteps[1:], torch.tensor([0], device=device)]))
        timesteps_one_start = reversed(self.scheduler.timesteps)

        self.scheduler.timesteps = timesteps_zero_start
        # self.scheduler.timesteps = timesteps_one_start

        timesteps = self.scheduler.timesteps

        latents_list = []
        latents_list.append(latents)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps * fixed_point_iterations) as progress_bar:
            for i, t in enumerate(timesteps):
                original_latents = latents.clone()
                for j in range(fixed_point_iterations):
                    if self.interrupt:
                        continue

                    if j == 0:
                        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        timestep = timesteps[i].expand(latents.shape[0]).to(latents.dtype)
                    else:
                        timestep = timesteps_one_start[i].expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = self.transformer(
                        hidden_states=latents,
                        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype

                    # noise_pred = -noise_pred
                    latents = self.scheduler.step(noise_pred, t, original_latents, return_dict=False, step_index=i)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    # if XLA_AVAILABLE:
                    #     xm.mark_step()
                
                latents_list.append(latents)

        # Offload all models
        self.maybe_free_model_hooks()

        return latents_list