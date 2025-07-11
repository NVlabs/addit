# Copyright (C) 2025 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import torch
import numpy as np
import torch.nn.functional as F
from skimage import filters
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, find_objects

def dilate_mask(latents_mask, k, latents_dtype):
    # Reshape the mask to 2D (64x64)
    mask_2d = latents_mask.view(64, 64)

    # Create a square kernel for dilation
    kernel = torch.ones(2*k+1, 2*k+1, device=mask_2d.device, dtype=mask_2d.dtype)

    # Add two dimensions to make it compatible with conv2d
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)

    # Perform dilation using conv2d
    dilated_mask = F.conv2d(mask_4d, kernel.unsqueeze(0).unsqueeze(0), padding=k)

    # Threshold the result to get a binary mask
    dilated_mask = (dilated_mask > 0).to(mask_2d.dtype)

    # Reshape back to the original shape and convert to the desired dtype
    dilated_mask = dilated_mask.view(4096, 1).to(latents_dtype)

    return dilated_mask

def clipseg_predict(model, processor, image, text, device):
    inputs = processor(text=text, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.unsqueeze(1)
        preds = torch.sigmoid(preds)

        otsu_thr = filters.threshold_otsu(preds.cpu().numpy())
        subject_mask = (preds > otsu_thr).float()

    return subject_mask

def grounding_sam_predict(model, processor, sam_predictor, image, text, device):
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    input_boxes = results[0]["boxes"].cpu().numpy()

    if input_boxes.shape[0] == 0:
        return torch.ones((64, 64), device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    subject_mask = torch.tensor(masks[0], device=device)

    return subject_mask

def mask_to_box_sam_predict(mask, sam_predictor, image, text, device):
    H, W = image.size

    # Resize clipseg mask to image size
    mask = F.interpolate(mask.view(1, 1, mask.shape[-2], mask.shape[-1]), size=(H, W), mode='bilinear').view(H, W)
    mask_indices = torch.nonzero(mask)
    top_left = mask_indices.min(dim=0)[0]
    bottom_right = mask_indices.max(dim=0)[0]
    
    # numpy shape [1,4]
    input_boxes = np.array([[top_left[1].item(), top_left[0].item(), bottom_right[1].item(), bottom_right[0].item()]])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=True,
        )

    # subject_mask = torch.tensor(masks[0], device=device)
    subject_mask = torch.tensor(np.max(masks, axis=0), device=device)

    return subject_mask, input_boxes[0]

def mask_to_mask_sam_predict(mask, sam_predictor, image, text, device):
    H, W = (256, 256)

    # Resize clipseg mask to image size
    mask = F.interpolate(mask.view(1, 1, mask.shape[-2], mask.shape[-1]), size=(H, W), mode='bilinear').view(1, H, W)
    mask_input = mask.float().cpu().numpy()    

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            mask_input=mask_input,
            multimask_output=False,
        )

    subject_mask = torch.tensor(masks[0], device=device)

    return subject_mask

def mask_to_points_sam_predict(mask, sam_predictor, image, text, device):
    H, W = image.size

    # Resize clipseg mask to image size
    mask = F.interpolate(mask.view(1, 1, mask.shape[-2], mask.shape[-1]), size=(H, W), mode='bilinear').view(H, W)
    mask_indices = torch.nonzero(mask)

    # Randomly sample 10 points from the mask
    n_points = 2
    point_coords = mask_indices[torch.randperm(mask_indices.shape[0])[:n_points]].float().cpu().numpy()
    point_labels = torch.ones((n_points,)).float().cpu().numpy()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    subject_mask = torch.tensor(masks[0], device=device)

    return subject_mask

def attention_to_points_sam_predict(subject_attention, subject_mask, sam_predictor, image, text, device):
    H, W = image.size

    # Resize clipseg mask to image size
    subject_attention = F.interpolate(subject_attention.view(1, 1, subject_attention.shape[-2], subject_attention.shape[-1]), size=(H, W), mode='bilinear').view(H, W)
    subject_mask = F.interpolate(subject_mask.view(1, 1, subject_mask.shape[-2], subject_mask.shape[-1]), size=(H, W), mode='bilinear').view(H, W)

    # Get mask_bbox
    subject_mask_indices = torch.nonzero(subject_mask)
    top_left = subject_mask_indices.min(dim=0)[0]
    bottom_right = subject_mask_indices.max(dim=0)[0]
    box_width = bottom_right[1] - top_left[1]
    box_height = bottom_right[0] - top_left[0]

    # Define the number of points and minimum distance between points
    n_points = 3
    max_thr = 0.35
    max_attention = torch.max(subject_attention)
    min_distance = max(box_width, box_height) // (n_points + 1)  # Adjust this value to control spread
    # min_distance = max(min_distance, 75)

    # Initialize list to store selected points
    selected_points = []

    # Create a copy of the attention map
    remaining_attention = subject_attention.clone()

    for _ in range(n_points):
        if remaining_attention.max() < max_thr * max_attention:
            break

        # Find the highest attention point
        point = torch.argmax(remaining_attention)
        y, x = torch.unravel_index(point, remaining_attention.shape)
        y, x = y.item(), x.item()
        
        # Add the point to our list
        selected_points.append((x, y))
        
        # Zero out the area around the selected point
        y_min = max(0, y - min_distance)
        y_max = min(H, y + min_distance + 1)
        x_min = max(0, x - min_distance)
        x_max = min(W, x + min_distance + 1)
        remaining_attention[y_min:y_max, x_min:x_max] = 0

    # Convert selected points to numpy array
    point_coords = np.array(selected_points)
    point_labels = np.ones(point_coords.shape[0], dtype=int)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    subject_mask = torch.tensor(masks[0], device=device)

    return subject_mask, point_coords

def sam_refine_step(mask, sam_predictor, image, device):
    mask_indices = torch.nonzero(mask)
    top_left = mask_indices.min(dim=0)[0]
    bottom_right = mask_indices.max(dim=0)[0]
    
    # numpy shape [1,4]
    input_boxes = np.array([[top_left[1].item(), top_left[0].item(), bottom_right[1].item(), bottom_right[0].item()]])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(image)
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=True,
        )

    # subject_mask = torch.tensor(masks[0], device=device)
    subject_mask = torch.tensor(np.max(masks, axis=0), device=device)

    return subject_mask, input_boxes[0]

