import hashlib
import math
import os
import urllib
import requests
import warnings

import torch
from tqdm import tqdm


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    # from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L895    """
    res = arr[timesteps].float()
    dims_to_append = len(broadcast_shape) - len(res.shape)
    return res[(...,) + (None,) * dims_to_append]


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    # from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L45
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


class ConsistencyDecoder:
    def __init__(self, device="cuda:0", download_root=os.path.expanduser("~/.cache/clip")):
        self.n_distilled_steps = 64
        download_target = _download("https://openaipublic.azureedge.net/diff-vae/c9cebd3132dd9c42936d803e33424145a748843c8f716c0814838bdc8a2fe7cb/decoder.pt", download_root)
        self.ckpt = torch.jit.load(download_target).to(device)
        self.device = device
        sigma_data = 0.5
        betas = betas_for_alpha_bar(
            1024, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        ).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)
        self.c_skip = (
            sqrt_recip_alphas_cumprod
            * sigma_data**2
            / (sigmas**2 + sigma_data**2)
        )
        self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
        self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

    @staticmethod
    def round_timesteps(
        timesteps, total_timesteps, n_distilled_steps, truncate_start=True
    ):
        with torch.no_grad():
            space = torch.div(total_timesteps, n_distilled_steps, rounding_mode="floor")
            rounded_timesteps = (
                torch.div(timesteps, space, rounding_mode="floor") + 1
            ) * space
            if truncate_start:
                rounded_timesteps[rounded_timesteps == total_timesteps] -= space
            else:
                rounded_timesteps[rounded_timesteps == total_timesteps] -= space
                rounded_timesteps[rounded_timesteps == 0] += space
            return rounded_timesteps

    @staticmethod
    def ldm_transform_latent(z, extra_scale_factor=1):
        channel_means = [0.38862467, 0.02253063, 0.07381133, -0.0171294]
        channel_stds = [0.9654121, 1.0440036, 0.76147926, 0.77022034]

        if len(z.shape) != 4:
            raise ValueError()

        z = z * 0.18215
        channels = [z[:, i] for i in range(z.shape[1])]

        channels = [
            extra_scale_factor * (c - channel_means[i]) / channel_stds[i]
            for i, c in enumerate(channels)
        ]
        return torch.stack(channels, dim=1)

    @torch.no_grad()
    def __call__(
        self,
        features: torch.Tensor,
        schedule=[1.0, 0.5],
    ):
        features = self.ldm_transform_latent(features)
        ts = self.round_timesteps(
            torch.arange(0, 1024),
            1024,
            self.n_distilled_steps,
            truncate_start=False,
        )
        shape = (
            features.size(0),
            3,
            8 * features.size(2),
            8 * features.size(3),
        )
        x_start = torch.zeros(shape, device=features.device, dtype=features.dtype)
        schedule_timesteps = [int((1024 - 1) * s) for s in schedule]
        for i in schedule_timesteps:
            t = ts[i].item()
            t_ = torch.tensor([t] * features.shape[0]).to(self.device)
            noise = torch.randn_like(x_start)
            x_start = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t_, x_start.shape)
                * x_start
                + _extract_into_tensor(
                    self.sqrt_one_minus_alphas_cumprod, t_, x_start.shape
                )
                * noise
            )
            c_in = _extract_into_tensor(self.c_in, t_, x_start.shape)
            model_output = self.ckpt(c_in * x_start, t_, features=features)
            B, C = x_start.shape[:2]
            model_output, _ = torch.split(model_output, C, dim=1)
            pred_xstart = (
                _extract_into_tensor(self.c_out, t_, x_start.shape) * model_output
                + _extract_into_tensor(self.c_skip, t_, x_start.shape) * x_start
            ).clamp(-1, 1)
            x_start = pred_xstart
        return x_start


def save_image(image, name):
    import numpy as np
    from PIL import Image

    image = image[0].cpu().numpy()
    image = (image + 1.0) * 127.5
    image = image.clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image.transpose(1, 2, 0))
    image.save(name)


def load_image(uri, size=None, center_crop=False):
    import numpy as np
    from PIL import Image

    if os.path.isfile(uri):
        # load image from local
        image = Image.open(uri)
    else:
        # load image by url
        image = Image.open(requests.get(uri, stream=True).raw)
    # handle case of grayscale and RGBA images
    image = image.convert("RGB")

    if center_crop:
        image = image.crop(
            (
                (image.width - min(image.width, image.height)) // 2,
                (image.height - min(image.width, image.height)) // 2,
                (image.width + min(image.width, image.height)) // 2,
                (image.height + min(image.width, image.height)) // 2,
            )
        )
    if size is not None:
        image = image.resize(size)
    image = torch.tensor(np.array(image).transpose(2, 0, 1)).unsqueeze(0).float()
    image = image / 127.5 - 1.0
    return image
