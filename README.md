# Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2411.07232-b31b1b.svg)](https://arxiv.org/abs/2411.07232)

### [[Project Website](https://research.nvidia.com/labs/par/addit/)]

> **Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models**<br>
> Yoad Tewel<sup>1,2</sup>, Rinon Gal<sup>1,2</sup>, Dvir Samuel<sup>3</sup>, Yuval Atzmon<sup>1</sup>, Lior Wolf<sup>2</sup>, Gal Chechik<sup>1</sup> <br>
> <sup>1</sup>NVIDIA, <sup>2</sup>Tel Aviv University, <sup>3</sup>Bar-Ilan University

![](https://research.nvidia.com/labs/par/addit/static/images/Teaser.png)

>**Abstract**: <br>
> Adding Object into images based on text instructions is a challenging task in semantic image editing, requiring a balance between preserving the original scene and seamlessly integrating the new object in a fitting location. Despite extensive efforts, existing models often struggle with this balance, particularly with finding a natural location for adding an object in complex scenes. We introduce Add-it, a training-free approach that extends diffusion models' attention mechanisms to incorporate information from three key sources: the scene image, the text prompt, and the generated image itself. Our weighted extended-attention mechanism maintains structural consistency and fine details while ensuring natural object placement. Without task-specific fine-tuning, Add-it achieves state-of-the-art results on both real and generated image insertion benchmarks, including our newly constructed "Additing Affordance Benchmark" for evaluating object placement plausibility, outperforming supervised methods. Human evaluations show that Add-it is preferred in over 80% of cases, and it also demonstrates improvements in various automated metrics.

## Description
This repo will contain the official code for our Add-it paper.

## News
- **2024 Nov 11**: The project page and arXiv paper for Add-it are now live. We are actively working on making the code available as soon as possible!

## TODO:
- [] Release code.

## Citation
If you make use of our work, please cite our paper:

```
@misc{tewel2024addit,
    title={Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models},
    author={Yoad Tewel and Rinon Gal and Dvir Samuel and Yuval Atzmon and Lior Wolf and Gal Chechik},
    year={2024},
    eprint={2411.07232},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```