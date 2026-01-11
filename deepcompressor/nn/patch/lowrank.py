# -*- coding: utf-8 -*-

import torch
import torch.linalg
import torch.nn as nn

from ...utils.hooks import AccumBranchHook, BaseInputPackager, BaseOutputPackager

__all__ = ["LowRankBranch"]


class LowRankBranch(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float = 1.0, weight: torch.Tensor | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        if rank == 0:
            self.a, self.b = None, None
        elif rank < 0:
            self.a, self.b = nn.Linear(in_features, out_features, bias=False), nn.Identity()
        else:
            self.a, self.b = nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features, bias=False)
        self.reset_parameters(weight)

    @torch.no_grad()
    def reset_parameters(self, weight: torch.Tensor | None = None) -> None:
        if weight is None:
            if self.rank < 0:
                nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                nn.init.kaiming_uniform_(self.a.weight)
                nn.init.zeros_(self.b.weight)
            return
        if weight.ndim >= 2:
            assert weight.shape[2:].numel() == 1, "LinearLoRAHook only supports 2D input tensor"
        weight = weight.view(weight.shape[0], -1)
        device, dtype = weight.device, weight.dtype
        self.to(device=device, dtype=dtype)
        out_features, in_features = weight.shape
        assert self.in_features == in_features, "Input features size mismatch"
        assert self.out_features == out_features, "Output features size mismatch"
        if self.rank < 0:
            self.a.weight.data.copy_(weight)
        elif self.rank > 0:
            # AntiGravity Fix: Use float32 SVD on the original device (GPU).
            # Tests confirm torch.linalg.svd on GPU with float32 takes ~2 seconds for 3072x12288.
            # double (float64) causes extreme slowness/hangs on consumer GPUs.
            # bfloat16 is not supported by svd.
            # Therefore, float32 is the correct, fast, and precise solution.
            weight_fp32 = weight.to(dtype=torch.float32) # Stay on device (GPU)
            u, s, vh = torch.linalg.svd(weight_fp32, full_matrices=False)
            
            # tensor: [oc, ic], u: [oc, rank], s: [rank], vh: [rank, ic] (if thin SVD and rank via slice)
            # Actually torch.linalg.svd returns u, s, Vh
            # s is 1D vector of singular values.
            
            # Keep top-k rank
            u = u[:, : self.rank]
            s = s[: self.rank]
            vh = vh[: self.rank, :]
            
            us = u * s.unsqueeze(0) # [oc, rank] * [1, rank] -> [oc, rank]
            # u * s means u[:, i] * s[i].
            us = u * s
            
            self.a.weight.data.copy_(vh.to(device=device, dtype=dtype))
            self.b.weight.data.copy_(us.to(device=device, dtype=dtype))
            
            del weight_fp32, u, s, vh, us
            # Explicit cleanup to prevent VRAM fragmentation/accumulation during heavy SVD loops
            gc.collect()
            torch.cuda.empty_cache()






    def get_effective_weight(self) -> torch.Tensor | None:
        if self.rank == 0:
            return None
        elif self.rank < 0:
            return self.a.weight
        else:
            return self.b.weight @ self.a.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor | None:
        if self.a is None:
            return None
        else:
            if input.ndim <= 3:
                return self.alpha * self.b(self.a(input))
            else:
                assert input.ndim == 4
                assert input.shape[-1] != self.in_features
                assert input.shape[1] == self.in_features
                # [B, C, H, W] -> [B, H, W, C] -> [B, H * W, C]
                B, C, H, W = input.shape
                input = input.permute(0, 2, 3, 1).reshape(B, H * W, C)
                output = self.alpha * self.b(self.a(input))
                # [B, H * W, C] -> [B, H, W, C] -> [B, C, H, W]
                output = output.reshape(B, H, W, -1).permute(0, 3, 1, 2)
                return output

    def as_hook(
        self,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ) -> AccumBranchHook:
        """Wrap the module as a branch hook.

        Args:
            input_packager (`BaseInputPackager` or `None`, *optional*, defaults to `None`):
                Input packager.
            output_packager (`BaseOutputPackager` or `None`, *optional*, defaults to `None`):
                Output packager.
        Returns:
            `AccumBranchHook`:
                The branch hook.
        """
        return AccumBranchHook(self, input_packager=input_packager, output_packager=output_packager)
