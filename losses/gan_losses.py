from __future__ import annotations

import torch


def gradient_penalty(discriminator, real, fake, y, rpm, order_template) -> torch.Tensor:
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, 1, device=real.device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    score, _ = discriminator(interp, y, rpm, order_template)
    grad = torch.autograd.grad(
        outputs=score,
        inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(bsz, -1)
    return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
