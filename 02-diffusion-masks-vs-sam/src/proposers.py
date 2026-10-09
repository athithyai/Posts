"""Three ways to propose 'every object in the image', with no labels at all.

diffusion  - my reimplementation of the Diffuse2Seg idea from its abstract (no
             code is public): point prompts on a grid, propagated through the
             self-attention of a frozen text-to-image diffusion model in an
             edge-preserving way. Details below; every design choice that the
             abstract does not pin down is mine and is listed in the post.
sam        - Segment Anything automatic mask generator (transformers pipeline).
felz       - Felzenszwalb graph segmentation at three scales: the no-deep-learning
             baseline, so "better than nothing" has a number.

Each proposer returns (masks: list of HxW bool arrays, scores: array), best first.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

SD_REPO = "sd2-community/stable-diffusion-2-1-base"   # mirror; stabilityai removed the original


# ── diffusion self-attention ────────────────────────────────────────────────
class _StoreSelfAttn:
    """Attention processor that also keeps the head-averaged attention matrix."""

    def __init__(self, bank):
        self.bank = bank

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        b, n, _ = hidden_states.shape
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(hidden_states))
        v = attn.head_to_batch_dim(attn.to_v(hidden_states))
        probs = attn.get_attention_scores(q, k, None)               # (b*heads, n, n)
        self.bank.setdefault(n, []).append(probs.view(b, -1, n, n).mean(1)[0].float())
        out = attn.batch_to_head_dim(torch.bmm(probs, v))
        return attn.to_out[1](attn.to_out[0](out))


class DiffusionAttention:
    def __init__(self, device="cuda", repo=SD_REPO):
        # Components, not the pipeline: only the UNet, VAE and an empty-prompt
        # embedding are needed, and it keeps diffusers' pipeline imports out.
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        f16 = torch.float16
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=f16).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet", torch_dtype=f16).to(device)
        self.sched = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")
        tokz = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
        enc = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=f16).to(device)
        tok = tokz([""], padding="max_length", max_length=tokz.model_max_length, return_tensors="pt")
        with torch.no_grad():
            self.null = enc(tok.input_ids.to(device))[0]
        del enc
        torch.cuda.empty_cache()
        self.device = device
        self.bank = {}
        procs = {}
        for name in self.unet.attn_processors:
            procs[name] = _StoreSelfAttn(self.bank) if name.endswith("attn1.processor") \
                else self.unet.attn_processors[name]
        self.unet.set_attn_processor(procs)

    @torch.no_grad()
    def affinity(self, img, t=300, seed=0):
        """(4096, 4096) row-stochastic affinity on the 64 x 64 latent grid."""
        x = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device, torch.float16) / 127.5 - 1
        lat = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor
        g = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(lat.shape, generator=g, device=self.device, dtype=lat.dtype)
        ts = torch.tensor([t], device=self.device)
        self.bank.clear()
        self.unet(self.sched.add_noise(lat, noise, ts), ts, encoder_hidden_states=self.null)
        a64 = torch.stack(self.bank[4096]).mean(0)
        a32 = torch.stack(self.bank[1024]).mean(0)                 # (1024, 1024)
        # Lift 32x32 attention onto the 64x64 grid: each query copies to its 4
        # children, each key spreads its mass over its 4 children.
        a32 = a32.view(32, 32, 32, 32)
        a32 = a32.repeat_interleave(2, 0).repeat_interleave(2, 1)
        a32 = a32.repeat_interleave(2, 2).repeat_interleave(2, 3) / 4
        a = 0.5 * a64 + 0.5 * a32.reshape(4096, 4096)
        self.bank.clear()
        return a


def guided_filter(guide, src, r=8, eps=1e-3):
    """He et al. guided filter, grey guide: edge-preserving upsampling."""
    I = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    box = lambda z: cv2.boxFilter(z, -1, (2 * r + 1, 2 * r + 1))
    mI, mp = box(I), box(src)
    a = (box(I * src) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def _component_with(mask, rc):
    lab, _ = ndimage.label(mask)
    v = lab[rc]
    return lab == v if v else None



def diffusion_masks(model, img, t=300, sigma=12.0, steps=(1, 2, 4), taus=(0.2, 0.35, 0.5),
                    grid=16, keep=1000, nms_iou=0.85, delta=0.08):
    a = model.affinity(img, t=t)
    dev = a.device
    if np.isfinite(sigma):
        lab = cv2.cvtColor(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
        c = torch.from_numpy(lab.reshape(-1, 3).astype(np.float32)).to(dev)
        a = a * torch.exp(-torch.cdist(c, c) ** 2 / (2 * sigma ** 2))
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)

    # Seeds: a grid of point prompts on the latent grid.
    s = 64 // grid
    pts = [(r, cidx) for r in range(s // 2, 64, s) for cidx in range(s // 2, 64, s)]
    seeds = torch.zeros(4096, len(pts), device=dev)
    for j, (r, cidx) in enumerate(pts):
        seeds[r * 64 + cidx, j] = 1
    # Heat flows from each seed: column j of (A^T)^k s_j is "who attends to the seed".
    heats = {}
    h = seeds
    at = a.T.contiguous()
    for k in range(1, max(steps) + 1):
        h = at @ h
        if k in steps:
            hk = h / h.amax(0, keepdim=True).clamp_min(1e-12)
            heats[k] = hk.T.reshape(len(pts), 64, 64).cpu().numpy()

    cands = []
    for k, hk in heats.items():
        for j, rc in enumerate(pts):
            hm = hk[j]
            for tau in taus:
                m = _component_with(hm >= tau, rc)
                if m is None or m.sum() < 4:
                    continue
                lo = _component_with(hm >= tau - delta, rc)
                hi = _component_with(hm >= tau + delta, rc)
                stab = (hi.sum() / lo.sum()) if (hi is not None and lo is not None and lo.sum()) else 0.0
                # A 2-cell blob is trivially "stable"; discount masks of a few cells.
                score = stab * (1 - np.exp(-m.sum() / 16))
                cands.append((score, m, j, k, tau))
    if not cands:
        return [], np.array([])
    cands.sort(key=lambda z: -z[0])

    # Greedy NMS with every pairwise IoU computed in one matmul on the GPU.
    flat = torch.from_numpy(np.stack([c[1].ravel() for c in cands])).to(dev, torch.float32)
    inter = flat @ flat.T
    area = flat.sum(1)
    iou = (inter / (area[:, None] + area[None] - inter).clamp_min(1)).cpu().numpy()
    alive = np.ones(len(cands), bool)
    kept = []
    for i in range(len(cands)):
        if not alive[i]:
            continue
        kept.append(cands[i])
        alive &= iou[i] < nms_iou
        if len(kept) >= keep:
            break

    H, W = img.shape[:2]
    masks, scores = [], []
    for stab, m, j, k, tau in kept:
        up = cv2.resize(heats[k][j].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        up = guided_filter(img, up)
        rc = (int((pts[j][0] + 0.5) * H / 64), int((pts[j][1] + 0.5) * W / 64))
        full = _component_with(up >= tau, rc)
        if full is None:
            full = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(full)
        scores.append(stab)
    return masks, np.array(scores)


# ── Segment Anything ────────────────────────────────────────────────────────
class SAM:
    def __init__(self, repo="facebook/sam-vit-huge", device=0):
        from transformers import pipeline
        self.gen = pipeline("mask-generation", model=repo, device=device)

    def __call__(self, img):
        from PIL import Image
        out = self.gen(Image.fromarray(img), points_per_batch=64)
        masks = [np.asarray(m, bool) for m in out["masks"]]
        scores = np.asarray([float(s) for s in out["scores"]])
        o = np.argsort(-scores)
        return [masks[i] for i in o], scores[o]


# ── classical baseline ──────────────────────────────────────────────────────
def felz_masks(img, scales=(800, 200, 50)):
    from skimage.segmentation import felzenszwalb
    masks, scores = [], []
    for rank, sc in enumerate(scales):            # coarse first: big regions rank highest
        seg = felzenszwalb(img, scale=sc, sigma=0.8, min_size=50)
        ids, counts = np.unique(seg, return_counts=True)
        for i in ids[np.argsort(-counts)]:
            masks.append(seg == i)
            scores.append(len(scales) - rank)
    return masks, np.array(scores, float)
