import os
from typing import List
from cog import BasePredictor, Input, Path
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)

MODEL_CACHE_DIR = "/src/cache"
os.environ["HF_HOME"] = os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        # Load depth-midas annotator
        from controlnet_aux.midas import MidasDetector
        self.annotator = MidasDetector.from_pretrained(
            "valhalla/t2i-adapter-depth-midas-sdxl",
            filename="dpt_large_384.pt",
            model_type="dpt_large"
        ).to("cuda")
        
        # Load SDXL components
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-depth-midas-sdxl",
            torch_dtype=torch.float16,
            varient="fp16"
        ).to("cuda")
        
        # Load pipeline
        self.pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            adapter=adapter,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Enable memory efficient attention
        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Input prompt",
            default="Modern interior design, 4k photo, highly detailed"
        ),
        negative_prompt: str = Input(
            description="What to avoid in the image",
            default="blurry, bad quality, distorted"
        ),
        num_steps: int = Input(
            description="Number of steps",
            ge=20,
            le=100,
            default=30
        ),
        guidance_scale: float = Input(
            description="Guidance scale",
            ge=1.0,
            le=20.0,
            default=7.5
        ),
        adapter_conditioning_scale: float = Input(
            description="Adapter conditioning scale",
            ge=0.5,
            le=2.0,
            default=1.0
        )
    ) -> List[Path]:
        """Run a single prediction"""
        # Process input image
        image = Image.open(image).convert("RGB")
        
        # Get depth map
        depth = self.annotator(
            image,
            detect_resolution=512,
            image_resolution=1024
        )
        
        # Generate output
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=depth,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            adapter_conditioning_scale=adapter_conditioning_scale
        )
        
        # Save outputs
        outputs = []
        
        # Save depth map
        depth_path = Path("depth_map.png")
        Image.fromarray(depth).save(depth_path)
        outputs.append(depth_path)
        
        # Save generated image
        output_path = Path("output.png")
        output.images[0].save(output_path)
        outputs.append(output_path)
        
        return outputs