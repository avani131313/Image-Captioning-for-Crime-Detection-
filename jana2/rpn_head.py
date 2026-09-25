"""Class-agnostic objectness head (FCOS-lite), vendored for inference.

Originally trained in the previous project against a frozen YOLO trunk.
~0.21M params. It asks only "is there an object here?" — there is no class
branch, because naming is CLIP's job.

Input : three feature maps (P3/P4/P5) from a YOLO neck, channels [64,128,256]
Output: boxes (normalised xyxy) + objectness scores in 0..1

Training code (build_targets, rpn_loss) is deliberately not copied — we are
not training this. Only the forward pass and proposal decode.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ObjectnessRPN(nn.Module):
    def __init__(self, in_channels: List[int], strides: List[int],
                 width: int = 128) -> None:
        super().__init__()
        self.strides = strides
        self.projs = nn.ModuleList(nn.Conv2d(c, width, 1) for c in in_channels)
        self.tower = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.ReLU(inplace=True))
        self.obj_head = nn.Conv2d(width, 1, 1)
        self.box_head = nn.Conv2d(width, 4, 1)

    def forward(self, feats: List[Tensor]) -> List[Tuple[Tensor, Tensor]]:
        out = []
        for proj, f in zip(self.projs, feats):
            t = self.tower(proj(f))
            out.append((self.obj_head(t), F.softplus(self.box_head(t))))
        return out

    @torch.no_grad()
    def proposals(self, feats: List[Tensor], img_size: int = 640,
                  pre_topk: int = 300, iou: float = 0.6,
                  topk: int = 64) -> Tuple[Tensor, Tensor]:
        """Returns (boxes [B,topk,4] normalised xyxy, scores [B,topk])."""
        from torchvision.ops import nms

        preds = self.forward(feats)
        B = feats[0].shape[0]
        all_boxes, all_scores = [], []

        for (obj, ltrb), stride in zip(preds, self.strides):
            _, _, H, W = obj.shape
            ys = (torch.arange(H, device=obj.device) + 0.5) * stride
            xs = (torch.arange(W, device=obj.device) + 0.5) * stride
            cy, cx = torch.meshgrid(ys, xs, indexing="ij")
            cx, cy = cx.reshape(-1), cy.reshape(-1)

            o = obj.reshape(B, -1).sigmoid()
            d = ltrb.reshape(B, 4, -1) * stride
            x1 = cx[None] - d[:, 0]
            y1 = cy[None] - d[:, 1]
            x2 = cx[None] + d[:, 2]
            y2 = cy[None] + d[:, 3]
            bx = torch.stack([x1, y1, x2, y2], -1) / img_size

            k = min(pre_topk, o.shape[1])
            sc, idx = o.topk(k, dim=1)
            all_scores.append(sc)
            all_boxes.append(torch.gather(bx, 1, idx[..., None].expand(-1, -1, 4)))

        scores = torch.cat(all_scores, 1)
        boxes = torch.cat(all_boxes, 1).clamp(0, 1)

        out_b = torch.zeros(B, topk, 4, device=boxes.device)
        out_s = torch.zeros(B, topk, device=boxes.device)
        for i in range(B):
            keep = nms(boxes[i] * img_size, scores[i], iou)[:topk]
            out_b[i, :len(keep)] = boxes[i][keep]
            out_s[i, :len(keep)] = scores[i][keep]
        return out_b, out_s


def load_rpn(ckpt_path, in_channels, strides, device):
    """Load rpn_v2 weights. Checkpoint layout: {'head': state_dict, ...}."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["head"] if "head" in ck else ck
    m = ObjectnessRPN(in_channels, strides).to(device).eval()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    info = {k: v for k, v in ck.items() if k != "head"} if isinstance(ck, dict) else {}
    print(f"[rpn] loaded {ckpt_path} in={in_channels} strides={strides} {info}")
    if missing or unexpected:
        print(f"[rpn] WARNING missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    return m
