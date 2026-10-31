import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

import clip
from PIL import Image

from utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small
from losses.ot_loss import sinkhorn, sinkhorn_divergence, cosine_cost


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class PatchEmbedMatmul(torch.nn.Module):
    """Drop-in replacement for a ViT patch-embedding Conv2d (kernel_size == stride, no padding) as patchify + matmul.

    Mathematically identical, but on Apple Silicon the MPS backward of a 16x16/stride-16 conv w.r.t. its
    input is >1000x slower than the matmul (1.7 s vs 1.4 ms for 8 images on an M5 Pro), and that gradient is
    exactly what CLIP guidance backpropagates through. Shares the conv's parameters.
    """

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        assert conv.kernel_size == conv.stride and conv.padding == (0, 0) and conv.groups == 1
        self.weight = conv.weight  # CLIP's VisionTransformer reads its dtype from conv1.weight
        self.bias = conv.bias
        self.patch = conv.kernel_size[0]

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch
        patches = x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // p) * (w // p), c * p * p)
        out = patches @ self.weight.reshape(self.weight.shape[0], -1).T
        if self.bias is not None:
            out = out + self.bias
        return out.transpose(1, 2).reshape(b, -1, h // p, w // p)


def load_clip(name, device):
    model, preprocess = clip.load(name, device=device)
    conv1 = getattr(getattr(model, "visual", None), "conv1", None)
    if torch.device(device).type == "mps" and isinstance(conv1, torch.nn.Conv2d) and conv1.kernel_size == conv1.stride:
        model.visual.conv1 = PatchEmbedMatmul(conv1)
    return model, preprocess


class CLIPLoss(torch.nn.Module):
    def __init__(self, device, lambda_direction=1., lambda_patch=0., lambda_global=0., lambda_manifold=0., lambda_texture=0., patch_loss_type='mae', direction_loss_type='cosine', clip_model='ViT-B/32',
                 lambda_ot=0., ot_eps=0.05, ot_n_views=8, ot_n_iters=100):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, clip_preprocess = load_clip(clip_model, self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        self.target_direction      = None
        self.patch_text_directions = None

        self.patch_loss     = DirectionLoss(patch_loss_type)
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)

        self.lambda_global    = lambda_global
        self.lambda_patch     = lambda_patch
        self.lambda_direction = lambda_direction
        self.lambda_manifold  = lambda_manifold
        self.lambda_texture   = lambda_texture

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()

        # RN50 is only used by the texture loss; loading it unconditionally costs ~250MB and seconds per run
        self.model_cnn = None
        if self.lambda_texture:
            self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
            self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                            preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
                                            preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor

        self.texture_loss = torch.nn.MSELoss()

        # Optimal-transport directional loss
        self.lambda_ot = lambda_ot
        self.ot_eps = ot_eps
        self.ot_n_views = ot_n_views
        self.ot_n_iters = ot_n_iters
        self.template_text_directions = None
        self.text_self_ot = None

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = image_features @ text_features.T

        return 1. - similarity
    
    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: list) -> torch.Tensor:
        with torch.no_grad():

            src_encoding = self.get_image_features(source_images)
            src_encoding = src_encoding.mean(dim=0, keepdim=True)

            target_encodings = []
            for target_img in target_images:
                preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
                
                encoding = self.model.encode_image(preprocessed)
                encoding /= encoding.norm(dim=-1, keepdim=True)

                target_encodings.append(encoding)
            
            target_encoding = torch.cat(target_encodings, axis=0)
            target_encoding = target_encoding.mean(dim=0, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.norm(dim=-1, keepdim=True)

        return direction

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = self.target_text_features @ self.src_text_features.T
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img).unsqueeze(2)
        target_img_features = self.get_image_features(target_img).unsqueeze(1)

        cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
        img_angle = torch.acos(cos_img_angle)

        text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
        cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

        return self.angle_loss(cos_img_angle, cos_text_angle)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]
            
    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        edit_direction = edit_direction / (edit_direction.norm(dim=-1, keepdim=True) + 1e-7)
        return self.direction_loss(edit_direction, self.target_direction).mean()

    def compute_template_text_directions(self, source_class: str, target_class: str) -> torch.Tensor:
        # one edit direction per prompt template ([T, d]) instead of their mean
        source_features = self.get_text_features(source_class).float()
        target_features = self.get_text_features(target_class).float()
        directions = target_features - source_features
        return directions / directions.norm(dim=-1, keepdim=True)

    def paired_views(self, src_img: torch.Tensor, target_img: torch.Tensor, n_views: int):
        """The full image plus n_views - 1 random resized crops/flips, applied identically to source and target
        so each view pair measures the edit rather than the augmentation. Returns two [n_views * B, C, H, W]."""
        src_views, target_views = [src_img], [target_img]
        size = src_img.shape[-2:]
        for _ in range(n_views - 1):
            i, j, h, w = transforms.RandomResizedCrop.get_params(src_img, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3))
            flip = torch.rand(()).item() < 0.5
            for img, views in ((src_img, src_views), (target_img, target_views)):
                v = TF.resized_crop(img, i, j, h, w, list(size), antialias=True)
                views.append(TF.hflip(v) if flip else v)
        return torch.cat(src_views), torch.cat(target_views)

    def clip_ot_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        """Sinkhorn divergence between the distribution of image edit directions (over augmented views) and
        the distribution of text edit directions (over prompt templates), with cost 1 - cos.

        The standard directional loss collapses both sides to a single mean vector; this matches the
        full distributions, so the edit has to agree with the set of template directions rather than
        with one averaged direction that no individual prompt actually produces."""
        if self.template_text_directions is None:
            self.template_text_directions = self.compute_template_text_directions(source_class, target_class)
            text = self.template_text_directions.unsqueeze(0)
            self.text_self_ot, _ = sinkhorn(cosine_cost(text, text), self.ot_eps, self.ot_n_iters)

        batch = src_img.shape[0]
        src_views, target_views = self.paired_views(src_img, target_img, self.ot_n_views)
        with torch.set_grad_enabled(src_img.requires_grad and torch.is_grad_enabled()):
            src_encoding = self.get_image_features(src_views)
        target_encoding = self.get_image_features(target_views)

        # OT in fp32: CLIP features are fp16 on GPU/MPS and exp(-C/eps) underflows in half precision
        directions = (target_encoding - src_encoding).float()
        directions = directions.reshape(self.ot_n_views, batch, -1).transpose(0, 1)  # [B, views, d]
        text = self.template_text_directions.unsqueeze(0).expand(batch, -1, -1)       # [B, templates, d]

        return sinkhorn_divergence(directions, text, self.ot_eps, self.ot_n_iters, yy=self.text_self_ot).mean()

    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
            
        tokens = clip.tokenize(text).to(self.device)
        image  = self.preprocess(img)

        logits_per_image, _ = self.model(image, tokens)

        return (1. - logits_per_image / 100).mean()

    def random_patch_centers(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape

        half_size = size // 2
        patch_centers = np.concatenate([np.random.randint(half_size, width - half_size,  size=(batch_size * num_patches, 1)),
                                        np.random.randint(half_size, height - half_size, size=(batch_size * num_patches, 1))], axis=1)

        return patch_centers

    def generate_patches(self, img: torch.Tensor, patch_centers, size):
        batch_size  = img.shape[0]
        num_patches = len(patch_centers) // batch_size
        half_size   = size // 2

        patches = []

        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):

                center_x = patch_centers[batch_idx * num_patches + patch_idx][0]
                center_y = patch_centers[batch_idx * num_patches + patch_idx][1]

                patch = img[batch_idx:batch_idx+1, :, center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

                patches.append(patch)

        patches = torch.cat(patches, axis=0)

        return patches

    def patch_scores(self, img: torch.Tensor, class_str: str, patch_centers, patch_size: int) -> torch.Tensor:

        parts = self.compose_text_with_templates(class_str, part_templates)    
        tokens = clip.tokenize(parts).to(self.device)
        text_features = self.encode_text(tokens).detach()

        patches        = self.generate_patches(img, patch_centers, patch_size)
        image_features = self.get_image_features(patches)

        similarity = image_features @ text_features.T

        return similarity

    def clip_patch_similarity(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        patch_size = 196 #TODO remove magic number

        patch_centers = self.random_patch_centers(src_img.shape, 4, patch_size) #TODO remove magic number
   
        src_scores    = self.patch_scores(src_img, source_class, patch_centers, patch_size)
        target_scores = self.patch_scores(target_img, target_class, patch_centers, patch_size)

        return self.patch_loss(src_scores, target_scores)

    def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.patch_text_directions is None:
            src_part_classes = self.compose_text_with_templates(source_class, part_templates)
            target_part_classes = self.compose_text_with_templates(target_class, part_templates)

            parts_classes = list(zip(src_part_classes, target_part_classes))

            self.patch_text_directions = torch.cat([self.compute_text_direction(pair[0], pair[1]) for pair in parts_classes], dim=0)

        patch_size = 510 # TODO remove magic numbers

        patch_centers = self.random_patch_centers(src_img.shape, 1, patch_size)

        patches = self.generate_patches(src_img, patch_centers, patch_size)
        src_features = self.get_image_features(patches)

        patches = self.generate_patches(target_img, patch_centers, patch_size)
        target_features = self.get_image_features(patches)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        cosine_dists = 1. - self.patch_direction_loss(edit_direction.unsqueeze(1), self.patch_text_directions.unsqueeze(0))

        patch_class_scores = cosine_dists * (edit_direction @ self.patch_text_directions.T).softmax(dim=-1)

        return patch_class_scores.mean()

    def cnn_feature_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_features = self.encode_images_with_cnn(src_img)
        target_features = self.encode_images_with_cnn(target_img)

        return self.texture_loss(src_features, target_features)

    def directional_log_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        """DiffusionCLIP's -log((2 - L) / 2) transform applied to each directional term separately.

        Each term lies in [0, 2], but a weighted sum of them need not, and log of a negative number is NaN."""
        def log_term(loss):
            return -torch.log((2 - loss).clamp_min(1e-6) / 2)

        loss = 0.0
        if self.lambda_direction:
            loss += self.lambda_direction * log_term(self.clip_directional_loss(src_img, source_class, target_img, target_class))
        if self.lambda_ot:
            loss += self.lambda_ot * log_term(self.clip_ot_directional_loss(src_img, source_class, target_img, target_class))
        return loss

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str, texture_image: torch.Tensor = None):
        clip_loss = 0.0

        if self.lambda_global:
            clip_loss += self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"])

        if self.lambda_patch:
            clip_loss += self.lambda_patch * self.patch_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_direction:
            clip_loss += self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_manifold:
            clip_loss += self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)

        if self.lambda_texture and (texture_image is not None):
            clip_loss += self.lambda_texture * self.cnn_feature_loss(texture_image, target_img)

        if self.lambda_ot:
            clip_loss += self.lambda_ot * self.clip_ot_directional_loss(src_img, source_class, target_img, target_class)

        return clip_loss
