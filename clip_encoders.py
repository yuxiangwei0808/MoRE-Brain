import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from torchvision import transforms
from transformers import (CLIPImageProcessor, CLIPModel, CLIPTextModel, 
                         CLIPTokenizer, CLIPVisionModel)
import open_clip


class CLIPTextEncoderDual(nn.Module):
    """
    Dual CLIP text encoder supporting both ViT-L and ViT-BigG encoders.
    
    Uses the text encoders from Stable Diffusion XL which includes both
    CLIP ViT-L (768d) and CLIP ViT-BigG (1280d) encoders.
    
    Args:
        truncation: Whether to use only EOS token embeddings
        output_mode: Output mode - 'joint', 'L', or 'BigG'
    """
    
    def __init__(self, truncation=False, output_mode='joint'):
        super().__init__()
        self.truncation = truncation
        self.output_mode = output_mode
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir='data/cache'
        )

        if output_mode in ['joint', 'L']:
            self.tokenizer_1 = pipeline.tokenizer
            self.text_encoder_1 = pipeline.text_encoder  # ViT-L
        if output_mode in ['joint', 'BigG']:
            self.tokenizer_2 = pipeline.tokenizer_2  # ViT-BigG
            self.text_encoder_2 = pipeline.text_encoder_2
        del pipeline
    
    @torch.no_grad()
    def forward(self, text):
        """
        Forward pass through the dual text encoders.
        
        Args:
            text: Input text string or list of strings
            
        Returns:
            Text embeddings based on output_mode configuration
        """
        if self.output_mode in ['joint', 'L']:
            device = self.text_encoder_1.device
            text1 = self.tokenizer_1(
                text, return_tensors="pt", padding="max_length", 
                truncation=True, max_length=self.tokenizer_1.model_max_length
            )
            input_1 = text1.input_ids.to(device)
            prompt_embeds_1 = self.text_encoder_1(input_1, output_hidden_states=True).hidden_states[-2]
            
        if self.output_mode in ['joint', 'BigG']:
            device = self.text_encoder_2.device
            text2 = self.tokenizer_2(
                text, return_tensors="pt", padding="max_length", 
                truncation=True, max_length=self.tokenizer_2.model_max_length
            )
            input_2 = text2.input_ids.to(device)
            prompt_embeds_2 = self.text_encoder_2(input_2, output_hidden_states=True).hidden_states[-2]

        if self.truncation:
            # Only keep EOS token embeddings; SOS token embeddings are consistent
            inputs = open_clip.tokenize(text)
            attention_mask = (inputs != 0).int()
            last_nonzero_idx = attention_mask.sum(dim=1) - 1
            if self.output_mode in ['joint', 'L']:
                eos_embeds_1 = prompt_embeds_1[torch.arange(prompt_embeds_1.shape[0]), last_nonzero_idx].unsqueeze(1)
            if self.output_mode in ['joint', 'BigG']:
                eos_embeds_2 = prompt_embeds_2[torch.arange(prompt_embeds_2.shape[0]), last_nonzero_idx].unsqueeze(1)
            # prompt_embeds_1 = torch.cat((prompt_embeds_1[:, :1], eos_embeds_1), dim=1)
            # prompt_embeds_2 = torch.cat((prompt_embeds_2[:, :1], eos_embeds_2), dim=1)
            
            if self.output_mode == 'joint':
                return torch.concat((eos_embeds_1, eos_embeds_2), dim=-1)
            elif self.output_mode == 'L':
                return eos_embeds_1
            else:
                return eos_embeds_2
        else:
            if self.output_mode == 'joint':
                return torch.concat((prompt_embeds_1, prompt_embeds_2), dim=-1)
            elif self.output_mode == 'L':
                return prompt_embeds_1
            else:
                return prompt_embeds_2


class CLIPImageEncoder(nn.Module):
    def __init__(self, version, is_proj, return_type='token', process_img=False):
        super().__init__()
        if version == 'BigG':
            version = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        elif version == 'L':
            version = "laion/CLIP-ViT-L-14-laion2B-S32B-b82k"

        self.return_type = return_type
        self.process_img = process_img
        if process_img:
            self.image_processor = CLIPImageProcessor.from_pretrained(version, cache_dir='data/cache')
    
        if is_proj:  # proj to 1280/768 dim
            # get projection head from CLIPModel
            clip_model = CLIPModel.from_pretrained(version, cache_dir='data/cache')
            self.image_encoder = clip_model.vision_model
            self.proj_head = clip_model.visual_projection
            del clip_model
        else:
            self.image_encoder = CLIPVisionModel.from_pretrained(version, cache_dir='data/cache')
            self.proj_head = nn.Identity()

    def forward(self, image):
        device = self.image_encoder.embeddings.patch_embedding.weight.device
        if self.process_img:
            image = self._process_image(image).to(device)

        output = self.image_encoder(image)
        tokens, pooled = output[0][:, 1:], output[1]  # remove cls token
        if self.return_type == 'both':
            return self.proj_head(tokens), self.proj_head(pooled).unsqueeze(1)
        elif self.return_type == 'pooled':
            return self.proj_head(pooled).unsqueeze(1)
        elif self.return_type == 'token':
            return self.proj_head(tokens)

    def _process_image(self, image):
        # Convert the batch of images to a list of PIL images
        pil_images = []
        for i in range(image.shape[0]):
            # Process each image individually
            img = image[i].permute(1, 2, 0).cpu().numpy()
            pil_img = transforms.ToPILImage()(img)
            pil_images.append(pil_img)
        
        # Process the batch of PIL images
        return self.image_processor(pil_images, return_tensors="pt").pixel_values