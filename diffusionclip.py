import time
import matplotlib
matplotlib.use("Agg")  # headless: only savefig is used
import matplotlib.pyplot as plt
import json
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import (
    DATASET_PATHS,
    MODEL_PATHS,
    HYBRID_MODEL_PATHS,
    HYBRID_CONFIG,
)
from datasets.imagenet_dic import IMAGENET_DIC

DDPM_CKPT_URLS = {
    "bedroom": "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt",
    "church_outdoor": "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt",
    "CelebA_HQ": "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt",
}


class SamplingAutocast(nn.DataParallel):
    """DataParallel (a pass-through on MPS/CPU) that runs the model under fp16/bf16 autocast when no gradients
    are needed, i.e. during inversion and sampling. Fine-tuning stays in fp32: fp16 activation gradients
    can underflow without a GradScaler. `.module` is still the bare UNet, so checkpoints are unchanged."""

    def __init__(self, module, amp_dtype=None):
        super().__init__(module)
        self.amp_dtype = amp_dtype

    def forward(self, *inputs, **kwargs):
        if self.amp_dtype is None or torch.is_grad_enabled():
            return super().forward(*inputs, **kwargs)
        device_type = next(self.module.parameters()).device.type
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
            return super().forward(*inputs, **kwargs).float()


def load_and_align(img_path, image_size, align_face):
    if align_face:
        try:
            # dlib is only needed for face alignment, so import lazily
            from utils.align_utils import run_alignment

            return run_alignment(img_path, output_size=image_size)
        except Exception:
            pass
    return Image.open(img_path).convert("RGB")


class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = getattr(config, "device", None) or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
            beta_schedule=getattr(config.diffusion, "beta_schedule", "linear"),
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == "fixedsmall":
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        # keep it on the device so denoising_step does not copy it from host memory at every step
        self.logvar = torch.from_numpy(self.logvar).float().to(self.device)

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]

    def load_model(self, model_path=None):
        """Builds the diffusion model for the configured dataset and loads `model_path` (or the
        pretrained checkpoint). Returns (model, state_dict, learn_sigma)."""
        dataset = self.config.data.dataset
        if dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if model_path:
                ckpt = torch.load(model_path, map_location="cpu")
            else:
                ckpt = torch.hub.load_state_dict_from_url(
                    DDPM_CKPT_URLS[self.config.data.category], map_location="cpu"
                )
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif dataset in ["FFHQ", "AFHQ", "IMAGENET", "TINY_IMAGENET"]:
            model = i_DDPM(dataset)
            ckpt = torch.load(model_path or MODEL_PATHS[dataset], map_location="cpu")
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            raise ValueError(f"Not implemented dataset {dataset}")
        model.load_state_dict(ckpt)
        model.to(self.device)
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(getattr(self.args, "amp", "none"))
        model = SamplingAutocast(model, amp_dtype)
        return model, ckpt, learn_sigma

    def clip_loss(self, lambda_direction=1, lambda_global=0):
        return CLIPLoss(
            self.device,
            lambda_direction=lambda_direction,
            lambda_patch=0,
            lambda_global=lambda_global,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name,
            lambda_ot=getattr(self.args, "lambda_ot", 0),
            ot_eps=getattr(self.args, "ot_eps", 0.05),
            ot_n_views=getattr(self.args, "ot_n_views", 8),
            ot_n_iters=getattr(self.args, "ot_n_iters", 100),
        )

    def clip_finetune(self):
        print(self.args.exp)
        print(f"   {self.src_txts}")
        print(f"-> {self.trg_txts}")

        # ----------- Model -----------#
        model, init_ckpt, learn_sigma = self.load_model(self.args.model_path)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(
            model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune
        )
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(
            optim_ft, step_size=1, gamma=self.args.sch_gamma
        )
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = self.clip_loss(lambda_direction=self.args.lambda_direction)
        # ArcFace weights are only shipped for faces; skip loading them when the ID loss is off
        id_loss_func = id_loss.IDLoss().to(self.device).eval() if self.args.id_loss_w > 0 else None

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ["train", "test"]:
            img_lat_pairs = []
            pairs_path = os.path.join(
                "precomputed/",
                f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
            )
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f"{mode} pairs exists")
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location="cpu")
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image(
                        (x0 + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder, f"{mode}_{step}_0_orig.png"
                        ),
                    )
                    tvu.save_image(
                        (x_id + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png",
                        ),
                    )
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(
                    self.config.data.dataset, DATASET_PATHS, self.config
                )
                loader_dic = get_dataloader(
                    train_dataset,
                    test_dataset,
                    bs_train=self.args.bs_train,
                    num_workers=self.config.data.num_workers,
                )
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image(
                    (x0 + 1) * 0.5,
                    os.path.join(self.args.image_folder, f"{mode}_{step}_0_orig.png"),
                )

                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_prev = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_prev,
                                models=model,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                            )

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image(
                        (x_lat + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png",
                        ),
                    )

                    with tqdm(
                        total=len(seq_inv), desc=f"Generative process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip(reversed((seq_inv)), reversed((seq_inv_next)))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_next = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                learn_sigma=learn_sigma,
                            )
                            progress_bar.update(1)

                    img_lat_pairs.append(
                        [x0, x.detach().clone(), x_lat.detach().clone()]
                    )
                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder,
                        f"{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png",
                    ),
                )
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join(
                "precomputed/",
                f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
            )
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
        )
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print("Uniform skip type")
        else:
            seq_train = list(range(self.args.t_0))
            print("No skip")
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None
            clip_loss_func.template_text_directions = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                save_name = (
                    f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                )
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f"{save_name} already exists.")
                        model.module.load_state_dict(torch.load(save_name, map_location="cpu"))
                        continue
                    else:
                        for step, (x0, x_id, x_lat) in enumerate(
                            img_lat_pairs_dic["train"]
                        ):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)

                            with tqdm(
                                total=len(seq_train), desc=f"CLIP iteration"
                            ) as progress_bar:
                                for t_it, (i, j) in enumerate(
                                    zip(reversed(seq_train), reversed(seq_train_next))
                                ):
                                    t = torch.full((x.shape[0],), float(i), device=self.device)
                                    t_next = torch.full((x.shape[0],), float(j), device=self.device)

                                    x = denoising_step(
                                        x,
                                        t=t,
                                        t_next=t_next,
                                        models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                    )

                                    progress_bar.update(1)

                            loss_clip = clip_loss_func.directional_log_loss(x0, src_txt, x, trg_txt)
                            loss_id = torch.mean(id_loss_func(x0, x)) if id_loss_func else torch.zeros((), device=self.device)
                            loss_l1 = nn.L1Loss()(x0, x)
                            loss = (
                                self.args.clip_loss_w * loss_clip
                                + self.args.id_loss_w * loss_id
                                + self.args.l1_loss_w * loss_l1
                            )
                            loss.backward()

                            optim_ft.step()
                            if not hasattr(self, 'loss_history'):
                                self.loss_history = {'clip': [], 'id': [], 'l1': [], 'total': []}
                            self.loss_history['clip'].append(loss_clip.item())
                            self.loss_history['id'].append(loss_id.item())
                            self.loss_history['l1'].append(loss_l1.item())
                            self.loss_history['total'].append(loss.item())
                            print(
                                f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}"
                            )

                            if self.args.save_train_image:
                                tvu.save_image(
                                    (x + 1) * 0.5,
                                    os.path.join(
                                        self.args.image_folder,
                                        f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png',
                                    ),
                                )
                            time_in_end = time.time()
                            print(
                                f"Training for 1 image takes {time_in_end - time_in_start:.4f}s"
                            )
                            if step == self.args.n_train_img - 1:
                                break

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f"Model {save_name} is saved.")
                        scheduler_ft.step()
                        if hasattr(self, 'loss_history') and len(self.loss_history['total']) > 0:
                            plt.figure()
                            plt.plot(self.loss_history['clip'], label='CLIP Loss')
                            if sum(self.loss_history['id']) > 0:
                                plt.plot(self.loss_history['id'], label='ID Loss')
                            plt.plot(self.loss_history['l1'], label='L1 Loss')
                            plt.plot(self.loss_history['total'], label='Total Loss')
                            plt.legend()
                            plt.xlabel('Steps')
                            plt.ylabel('Loss')
                            plt.title('Training Losses')
                            plot_path = os.path.join(self.args.image_folder, f'training_loss.png')
                            plt.savefig(plot_path)
                            plt.close()
                            with open(os.path.join(self.args.image_folder, f'training_loss.json'), 'w') as f:
                                json.dump(self.loss_history, f)

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name, map_location="cpu"))

                    model.eval()
                    mode = "test"
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat.to(self.device)
                            with tqdm(
                                total=len(seq_test), desc=f"Eval iteration"
                            ) as progress_bar:
                                for i, j in zip(
                                    reversed(seq_test), reversed(seq_test_next)
                                ):
                                    t = torch.full((x.shape[0],), float(i), device=self.device)
                                    t_next = torch.full((x.shape[0],), float(j), device=self.device)

                                    x = denoising_step(
                                        x,
                                        t=t,
                                        t_next=t_next,
                                        models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                    )

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image(
                                (x + 1) * 0.5,
                                os.path.join(
                                    self.args.image_folder,
                                    f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png',
                                ),
                            )
                            if step == self.args.n_test_img - 1:
                                break

    def clip_finetune_eff(self):
        print(self.args.exp)
        print(f"   {self.src_txts}")
        print(f"-> {self.trg_txts}")

        # ----------- Model -----------#
        model, init_ckpt, learn_sigma = self.load_model(self.args.model_path)

        # ----------- Optimizer and Scheduler -----------#
        print(f"Setting optimizer with lr={self.args.lr_clip_finetune}")
        optim_ft = torch.optim.Adam(
            model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune
        )
        # optim_ft = torch.optim.SGD(model.parameters(), weight_decay=0, lr=self.args.lr_clip_finetune)#, momentum=0.9)
        init_opt_ckpt = optim_ft.state_dict()
        scheduler_ft = torch.optim.lr_scheduler.StepLR(
            optim_ft, step_size=1, gamma=self.args.sch_gamma
        )
        init_sch_ckpt = scheduler_ft.state_dict()

        # ----------- Loss -----------#
        print("Loading losses")
        clip_loss_func = self.clip_loss(lambda_direction=self.args.lambda_direction)
        if self.config.data.dataset == "CelebA_HQ" and self.args.id_loss_w > 0:
            id_loss_func = id_loss.IDLoss().to(self.device).eval()
        else:
            id_loss_func = None

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        img_lat_pairs_dic = {}

        for mode in ["train", "test"]:
            img_lat_pairs = []
            if self.args.edit_attr in ["female", "male"]:
                self.config.data.dataset = "GENDER"
                self.config.data.category = "GENDER"
                if self.args.edit_attr == "female":
                    pairs_path = os.path.join(
                        "precomputed/",
                        f"{self.config.data.category}_male_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
                    )
                else:
                    pairs_path = os.path.join(
                        "precomputed/",
                        f"{self.config.data.category}_female_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
                    )

            elif self.config.data.dataset == "IMAGENET":
                if self.args.target_class_num is not None:
                    pairs_path = os.path.join(
                        "precomputed/",
                        f"{self.config.data.category}_{IMAGENET_DIC[str(self.args.target_class_num)][1]}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
                    )
                else:
                    pairs_path = os.path.join(
                        "precomputed/",
                        f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
                    )

            else:
                pairs_path = os.path.join(
                    "precomputed/",
                    f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
                )
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f"{mode} pairs exists")
                img_lat_pairs_dic[mode] = torch.load(
                    pairs_path, map_location=torch.device("cpu")
                )
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image(
                        (x0 + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder, f"{mode}_{step}_0_orig.png"
                        ),
                    )
                    tvu.save_image(
                        (x_id + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png",
                        ),
                    )
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                if self.args.edit_attr == "female":
                    train_dataset, test_dataset = get_dataset(
                        self.config.data.dataset,
                        DATASET_PATHS,
                        self.config,
                        gender="male",
                    )
                elif self.args.edit_attr == "male":
                    train_dataset, test_dataset = get_dataset(
                        self.config.data.dataset,
                        DATASET_PATHS,
                        self.config,
                        gender="female",
                    )
                else:
                    train_dataset, test_dataset = get_dataset(
                        self.config.data.dataset,
                        DATASET_PATHS,
                        self.config,
                        target_class_num=self.args.target_class_num,
                    )

                loader_dic = get_dataloader(
                    train_dataset,
                    test_dataset,
                    bs_train=self.args.bs_train,
                    num_workers=self.config.data.num_workers,
                )
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image(
                    (x0 + 1) * 0.5,
                    os.path.join(self.args.image_folder, f"{mode}_{step}_0_orig.png"),
                )

                x = x0.clone()
                model.eval()
                time_s = time.time()
                with torch.no_grad():
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_prev = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_prev,
                                models=model,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                            )

                            progress_bar.update(1)
                    time_e = time.time()
                    print(f"{time_e - time_s} seconds")
                    x_lat = x.clone()
                    tvu.save_image(
                        (x_lat + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png",
                        ),
                    )

                    with tqdm(
                        total=len(seq_inv), desc=f"Generative process {mode} {step}"
                    ) as progress_bar:
                        time_s = time.time()
                        for it, (i, j) in enumerate(
                            zip(reversed((seq_inv)), reversed((seq_inv_next)))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_next = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                learn_sigma=learn_sigma,
                            )
                            progress_bar.update(1)
                        time_e = time.time()
                        print(f"{time_e - time_s} seconds")

                    img_lat_pairs.append(
                        [x0, x.detach().clone(), x_lat.detach().clone()]
                    )
                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder,
                        f"{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png",
                    ),
                )
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            # pairs_path = os.path.join('precomputed/',
            #                           f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
        )
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print("Uniform skip type")
        else:
            seq_train = list(range(self.args.t_0))
            print("No skip")
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None
            clip_loss_func.template_text_directions = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                save_name = (
                    f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                )
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f"{save_name} already exists.")
                        model.module.load_state_dict(torch.load(save_name, map_location="cpu"))
                        continue
                    else:
                        for step, (x0, _, x_lat) in enumerate(
                            img_lat_pairs_dic["train"]
                        ):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(
                                total=len(seq_train), desc=f"CLIP iteration"
                            ) as progress_bar:
                                for t_it, (i, j) in enumerate(
                                    zip(reversed(seq_train), reversed(seq_train_next))
                                ):
                                    t = torch.full((x.shape[0],), float(i), device=self.device)
                                    t_next = torch.full((x.shape[0],), float(j), device=self.device)

                                    x, x0_t = denoising_step(
                                        x,
                                        t=t,
                                        t_next=t_next,
                                        models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                        out_x0_t=True,
                                    )

                                    progress_bar.update(1)
                                    x = x.detach().clone()

                                    loss_clip = clip_loss_func.directional_log_loss(x0, src_txt, x0_t, trg_txt)
                                    loss_l1 = nn.L1Loss()(x0, x0_t)
                                    loss = (
                                        self.args.clip_loss_w * loss_clip
                                        + self.args.l1_loss_w * loss_l1
                                    )
                                    if id_loss_func is not None:
                                        loss_id = torch.mean(id_loss_func(x0, x0_t))
                                        loss += self.args.id_loss_w * loss_id
                                    loss.backward()

                                    optim_ft.step()
                                    for p in model.module.parameters():
                                        p.grad = None
                                    if not hasattr(self, 'loss_history'):
                                        self.loss_history = {'clip': [], 'id': [], 'l1': [], 'total': []}
                                    self.loss_history['clip'].append(loss_clip.item())
                                    if id_loss_func is not None:
                                        self.loss_history['id'].append(loss_id.item())
                                    else:
                                        self.loss_history['id'].append(0)
                                    self.loss_history['l1'].append(loss_l1.item())
                                    self.loss_history['total'].append(loss.item())
                                    print(
                                        f"CLIP {step}-{it_out}: loss_clip: {loss_clip:.3f}"
                                    )
                                    # break

                            if self.args.save_train_image:
                                tvu.save_image(
                                    (x0_t + 1) * 0.5,
                                    os.path.join(
                                        self.args.image_folder,
                                        f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png',
                                    ),
                                )
                            time_in_end = time.time()
                            print(
                                f"Training for 1 image takes {time_in_end - time_in_start:.4f}s"
                            )
                            if step == self.args.n_train_img - 1:
                                break

                        if isinstance(model, nn.DataParallel):
                            torch.save(model.module.state_dict(), save_name)
                        else:
                            torch.save(model.state_dict(), save_name)
                        print(f"Model {save_name} is saved.")
                        scheduler_ft.step()
                        if hasattr(self, 'loss_history') and len(self.loss_history['total']) > 0:
                            plt.figure()
                            plt.plot(self.loss_history['clip'], label='CLIP Loss')
                            if sum(self.loss_history['id']) > 0:
                                plt.plot(self.loss_history['id'], label='ID Loss')
                            plt.plot(self.loss_history['l1'], label='L1 Loss')
                            plt.plot(self.loss_history['total'], label='Total Loss')
                            plt.legend()
                            plt.xlabel('Steps')
                            plt.ylabel('Loss')
                            plt.title('Training Losses')
                            plot_path = os.path.join(self.args.image_folder, f'training_loss.png')
                            plt.savefig(plot_path)
                            plt.close()
                            with open(os.path.join(self.args.image_folder, f'training_loss.json'), 'w') as f:
                                json.dump(self.loss_history, f)

                # ----------- Eval -----------#
                if self.args.do_test:
                    if not self.args.do_train:
                        print(save_name)
                        model.module.load_state_dict(torch.load(save_name, map_location="cpu"))

                    model.eval()
                    mode = "test"
                    img_lat_pairs = img_lat_pairs_dic[mode]
                    for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):
                        with torch.no_grad():
                            x = x_lat.clone().to(self.device)
                            x0 = x0.to(self.device)
                            with tqdm(
                                total=len(seq_test), desc=f"Eval iteration"
                            ) as progress_bar:
                                for i, j in zip(
                                    reversed(seq_test), reversed(seq_test_next)
                                ):
                                    t = torch.full((x.shape[0],), float(i), device=self.device)
                                    t_next = torch.full((x.shape[0],), float(j), device=self.device)

                                    x = denoising_step(
                                        x,
                                        t=t,
                                        t_next=t_next,
                                        models=model,
                                        logvars=self.logvar,
                                        sampling_type=self.args.sample_type,
                                        b=self.betas,
                                        eta=self.args.eta,
                                        learn_sigma=learn_sigma,
                                    )

                                    progress_bar.update(1)

                            print(f"Eval {step}-{it_out}")
                            tvu.save_image(
                                (x + 1) * 0.5,
                                os.path.join(
                                    self.args.image_folder,
                                    f'{mode}_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_test_step}.png',
                                ),
                            )
                            if step == self.args.n_test_img - 1:
                                break

    def clip_latent_optim(self):
        # ----------- Data -----------#
        n = 1

        img = load_and_align(
            self.args.img_path,
            self.config.data.image_size,
            self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"],
        )
        img = img.resize(
            (self.config.data.image_size, self.config.data.image_size), Image.LANCZOS
        )
        img = np.array(img) / 255
        img = (
            torch.from_numpy(img)
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
            .unsqueeze(dim=0)
        )
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f"0_orig.png"))
        x0 = (img - 0.5) * 2.0

        # ----------- Model -----------#
        model, _, learn_sigma = self.load_model(self.args.model_path)
        model.eval()

        # ----------- Loss -----------#
        print("Loading losses")
        # ArcFace weights are only shipped for faces; skip loading them when the ID loss is off
        id_loss_func = id_loss.IDLoss().to(self.device).eval() if self.args.id_loss_w > 0 else None
        clip_loss_func = self.clip_loss(lambda_direction=0, lambda_global=1)

        # ----------- Invert Image to Latent -----------#
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        print(f"Finding latent")
        with torch.no_grad():
            x = x0.clone()

            with tqdm(total=len(seq_inv), desc=f"Inversion process") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = torch.full((x.shape[0],), float(i), device=self.device)
                    t_prev = torch.full((x.shape[0],), float(j), device=self.device)

                    x = denoising_step(
                        x,
                        t=t,
                        t_next=t_prev,
                        models=model,
                        logvars=self.logvar,
                        sampling_type="ddim",
                        b=self.betas,
                        eta=0,
                        learn_sigma=learn_sigma,
                    )

                    progress_bar.update(1)

            x_lat0 = x.clone()
            tvu.save_image(
                (x_lat0 + 1) * 0.5,
                os.path.join(
                    self.args.image_folder, f"1_lat_ninv{self.args.n_inv_step}.png"
                ),
            )

        # ----------- Latent Optimization -----------#
        print(f"CLIP loss latent optimization")
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
        )
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print("Uniform skip type")
        else:
            seq_train = list(range(self.args.t_0))
            print("No skip")
        seq_train_next = [-1] + list(seq_train[:-1])

        for txt_idx, (src_txt, trg_txt) in enumerate(zip(self.src_txts, self.trg_txts)):
            x_lat = nn.Parameter(x_lat0.clone())
            optimizer = torch.optim.Adam(
                [x_lat], weight_decay=0, lr=self.args.lr_clip_lat_opt
            )
            clip_loss_func.target_direction = None
            clip_loss_func.template_text_directions = None

            with torch.set_grad_enabled(True):
                for it in range(self.args.n_iter):
                    x = x_lat
                    optimizer.zero_grad()
                    with tqdm(
                        total=len(seq_train), desc=f"Generative process {trg_txt}-{it}"
                    ) as progress_bar:
                        for i, j in zip(reversed(seq_train), reversed(seq_train_next)):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_next = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=model,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                eta=self.args.eta,
                                learn_sigma=learn_sigma,
                            )
                            progress_bar.update(1)

                    loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                    loss_clip = -torch.log(loss_clip)
                    loss_id = torch.mean(id_loss_func(x0, x)) if id_loss_func else torch.zeros((), device=self.device)
                    loss_l1 = nn.L1Loss()(x0, x)

                    loss = (
                        self.args.clip_loss_w * loss_clip
                        + self.args.id_loss_w * loss_id
                        + self.args.l1_loss_w * loss_l1
                    )
                    loss.backward()
                    print(
                        f"CLIP opt: loss_clip: {loss_clip:.3f}, loss_id: {loss_id:.3f}, loss_l1: {loss_l1:.3f}"
                    )
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f'2_clip_{trg_txt.replace(" ", "_")}_t{self.args.t_0}_{it}_ngen{self.args.n_train_step}.png',
                        ),
                    )

                    optimizer.step()

    def edit_images_from_dataset(self):
        # ----------- Models -----------#
        print(self.args.exp)
        models = []
        model_paths = [None, self.args.model_path]
        for model_path in model_paths:
            model_i, _, learn_sigma = self.load_model(model_path)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        # ----------- Precompute Latents thorugh Inversion Process -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        n = 1
        img_lat_pairs_dic = {}
        for mode in ["test"]:
            img_lat_pairs = []
            pairs_path = os.path.join(
                "precomputed/",
                f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
            )

            if os.path.exists(pairs_path):
                print(f"{mode} pairs exists")
                img_lat_pairs_dic[mode] = torch.load(pairs_path, map_location="cpu")
                for step, (x0, x_id, e_id) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image(
                        (x0 + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder, f"{mode}_{step}_0_orig.png"
                        ),
                    )
                    tvu.save_image(
                        (x_id + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder, f"{mode}_{step}_1_rec.png"
                        ),
                    )
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(
                    self.config.data.dataset, DATASET_PATHS, self.config
                )
                loader_dic = get_dataloader(
                    train_dataset,
                    test_dataset,
                    bs_train=self.args.bs_train,
                    num_workers=self.config.data.num_workers,
                )
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image(
                    (x0 + 1) * 0.5,
                    os.path.join(self.args.image_folder, f"{mode}_{step}_0_orig.png"),
                )

                x = x0.clone()
                with torch.no_grad():
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_prev = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_prev,
                                models=models,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                                ratio=0,
                            )

                            progress_bar.update(1)

                    x_lat = x.clone()
                    tvu.save_image(
                        (x_lat + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png",
                        ),
                    )

                    with tqdm(
                        total=len(seq_inv), desc=f"Generative process {mode} {step}"
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip(reversed((seq_inv)), reversed((seq_inv_next)))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_next = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=models,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                eta=self.args.eta,
                                learn_sigma=learn_sigma,
                                ratio=0,
                            )

                            progress_bar.update(1)

                    img_lat_pairs.append(
                        [x0, x.detach().clone(), x_lat.detach().clone()]
                    )

                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(self.args.image_folder, f"{mode}_{step}_1_rec.png"),
                )
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join(
                "precomputed/",
                f"{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth",
            )
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Generative Process -----------#
        print(
            f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
        )
        if self.args.n_test_step != 0:
            seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
            seq_test = [int(s) for s in list(seq_test)]
            print("Uniform skip type")
        else:
            seq_test = list(range(self.args.t_0))
            print("No skip")
        seq_test_next = [-1] + list(seq_test[:-1])
        print("Start evaluation")
        eval_modes = ["test"]
        for mode in eval_modes:

            img_lat_pairs = img_lat_pairs_dic[mode]
            for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs):

                with torch.no_grad():
                    x = x_lat.to(self.device)
                    with tqdm(
                        total=len(seq_test), desc=f"Eval iteration"
                    ) as progress_bar:
                        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_next = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_next,
                                models=models,
                                logvars=self.logvar,
                                sampling_type=self.args.sample_type,
                                b=self.betas,
                                eta=self.args.eta,
                                learn_sigma=learn_sigma,
                                ratio=self.args.model_ratio,
                                hybrid=self.args.hybrid_noise,
                                hybrid_config=HYBRID_CONFIG,
                            )

                            progress_bar.update(1)

                    print(f"Eval {step}")
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"{mode}_{step}_2_clip_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png",
                        ),
                    )

    def edit_one_image(self):
        # ----------- Data -----------#
        n = self.args.bs_test

        img = load_and_align(
            self.args.img_path,
            self.config.data.image_size,
            self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"],
        )
        img = img.resize(
            (self.config.data.image_size, self.config.data.image_size), Image.LANCZOS
        )
        img = np.array(img) / 255
        img = (
            torch.from_numpy(img)
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
            .unsqueeze(dim=0)
            .repeat(n, 1, 1, 1)
        )
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f"0_orig.png"))
        x0 = (img - 0.5) * 2.0

        # ----------- Models -----------#
        models = []

        if self.args.hybrid_noise:
            model_paths = [None] + HYBRID_MODEL_PATHS
        else:
            model_paths = [None, self.args.model_path]

        for model_path in model_paths:
            model_i, _, learn_sigma = self.load_model(model_path)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            # ---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(
                    self.args.image_folder,
                    f"x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth",
                )
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(
                        total=len(seq_inv), desc=f"Inversion process "
                    ) as progress_bar:
                        for it, (i, j) in enumerate(
                            zip((seq_inv_next[1:]), (seq_inv[1:]))
                        ):
                            t = torch.full((x.shape[0],), float(i), device=self.device)
                            t_prev = torch.full((x.shape[0],), float(j), device=self.device)

                            x = denoising_step(
                                x,
                                t=t,
                                t_next=t_prev,
                                models=models,
                                logvars=self.logvar,
                                sampling_type="ddim",
                                b=self.betas,
                                eta=0,
                                learn_sigma=learn_sigma,
                                ratio=0,
                            )

                            progress_bar.update(1)
                        x_lat = x.clone()
                        torch.save(x_lat, x_lat_path)
                else:
                    print("Latent exists.")
                    x_lat = torch.load(x_lat_path, map_location=self.device)

            # ----------- Generative Process -----------#
            print(
                f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                f" Steps: {self.args.n_test_step}/{self.args.t_0}"
            )
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print("Uniform skip type")
            else:
                seq_test = list(range(self.args.t_0))
                print("No skip")
            seq_test_next = [-1] + list(seq_test[:-1])

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = (
                        x0 * a[self.args.t_0 - 1].sqrt()
                        + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                    )
                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder, f"1_lat_ninv{self.args.n_inv_step}.png"
                    ),
                )

                with tqdm(
                    total=len(seq_test), desc="Generative process {}".format(it)
                ) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = torch.full((x.shape[0],), float(i), device=self.device)
                        t_next = torch.full((x.shape[0],), float(j), device=self.device)

                        x = denoising_step(
                            x,
                            t=t,
                            t_next=t_next,
                            models=models,
                            logvars=self.logvar,
                            sampling_type=self.args.sample_type,
                            b=self.betas,
                            eta=self.args.eta,
                            learn_sigma=learn_sigma,
                            ratio=self.args.model_ratio,
                            hybrid=self.args.hybrid_noise,
                            hybrid_config=HYBRID_CONFIG,
                        )

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image(
                                (x + 1) * 0.5,
                                os.path.join(
                                    self.args.image_folder,
                                    f"2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png",
                                ),
                            )
                        progress_bar.update(1)

                x0 = x.clone()
                if self.args.model_path:
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}_{self.args.model_path.split('/')[-1].replace('.pth','')}.png",
                        ),
                    )
                else:
                    tvu.save_image(
                        (x + 1) * 0.5,
                        os.path.join(
                            self.args.image_folder,
                            f"3_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png",
                        ),
                    )

    def unseen2unseen(self):
        # ----------- Data -----------#
        n = self.args.bs_test
        # if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
        #     try:
        #         img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
        #     except:
        #         img = Image.open(self.args.img_path).convert("RGB")
        # else:
        img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize(
            (self.config.data.image_size, self.config.data.image_size), Image.LANCZOS
        )
        img = np.array(img) / 255
        img = (
            torch.from_numpy(img)
            .type(torch.FloatTensor)
            .permute(2, 0, 1)
            .unsqueeze(dim=0)
        )
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f"0_orig.png"))
        img = img.repeat(n, 1, 1, 1)
        x0 = (img - 0.5) * 2.0

        # ----------- Models -----------#
        models = []
        model_paths = [None, self.args.model_path]
        for model_path in model_paths:
            model_i, _, learn_sigma = self.load_model(model_path)
            model_i.eval()
            print(f"{model_path} is loaded.")
            models.append(model_i)

        with torch.no_grad():
            print(
                f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}"
            )
            print("Uniform skip type")
            for it in range(self.args.n_iter):
                seq_test = np.linspace(0, 1, 6) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                seq_test_next = [-1] + list(seq_test[:-1])

                # ----------- Stochastic Foward Process -----------#
                e = torch.randn_like(x0)
                a = (1 - self.betas).cumprod(dim=0)
                x = (
                    x0 * a[self.args.t_0 - 1].sqrt()
                    + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                )
                tvu.save_image(
                    (x + 1) * 0.5, os.path.join(self.args.image_folder, f"1_lat.png")
                )

                # ----------- Generative Process -----------#
                with tqdm(
                    total=len(seq_test), desc="Generative process {}".format(it)
                ) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = torch.full((x.shape[0],), float(i), device=self.device)
                        t_next = torch.full((x.shape[0],), float(j), device=self.device)

                        x = denoising_step(
                            x,
                            t=t,
                            t_next=t_next,
                            models=models,
                            logvars=self.logvar,
                            sampling_type=self.args.sample_type,
                            b=self.betas,
                            eta=0,
                            learn_sigma=learn_sigma,
                            ratio=0,
                        )

                        progress_bar.update(1)

                x0 = x.clone()

                tvu.save_image(
                    (x + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder,
                        f"2_gen_t{self.args.t_0}_it{it}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}.png",
                    ),
                )

            # ---------------- Invert Image to Latent through Detriministic Process -------------------#
            seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
            seq_inv = [int(s) for s in list(seq_inv)]
            seq_inv_next = [-1] + list(seq_inv[:-1])

            x = x0.clone()
            with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                    t = torch.full((x.shape[0],), float(i), device=self.device)
                    t_prev = torch.full((x.shape[0],), float(j), device=self.device)

                    x = denoising_step(
                        x,
                        t=t,
                        t_next=t_prev,
                        models=models,
                        logvars=self.logvar,
                        sampling_type="ddim",
                        b=self.betas,
                        eta=0,
                        learn_sigma=learn_sigma,
                        ratio=0,
                    )

                    progress_bar.update(1)
                x_lat = x.clone()
                tvu.save_image(
                    (x_lat + 1) * 0.5,
                    os.path.join(
                        self.args.image_folder, f"3_lat_ninv{self.args.n_inv_step}.png"
                    ),
                )

            # ----------- Generative Process with Finetuned Model -----------#
            print(
                f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                f" Steps: {self.args.n_test_step}/{self.args.t_0}"
            )
            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print("Uniform skip type")
            else:
                seq_test = list(range(self.args.t_0))
                print("No skip")
            seq_test_next = [-1] + list(seq_test[:-1])

            x = x_lat.clone()

            with tqdm(
                total=len(seq_test), desc="Generative process {}".format(it)
            ) as progress_bar:
                for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                    t = torch.full((x.shape[0],), float(i), device=self.device)
                    t_next = torch.full((x.shape[0],), float(j), device=self.device)

                    x = denoising_step(
                        x,
                        t=t,
                        t_next=t_next,
                        models=models,
                        logvars=self.logvar,
                        sampling_type=self.args.sample_type,
                        b=self.betas,
                        eta=self.args.eta,
                        learn_sigma=learn_sigma,
                        ratio=self.args.model_ratio,
                    )

                    progress_bar.update(1)

            tvu.save_image(
                (x + 1) * 0.5,
                os.path.join(
                    self.args.image_folder,
                    f"4_gen_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_mrat{self.args.model_ratio}.png",
                ),
            )
