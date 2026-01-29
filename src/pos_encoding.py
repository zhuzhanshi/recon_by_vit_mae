import torch


def linear_3ch(
    gray: torch.Tensor,
    series: int,
    dx: int,
    dy: int,
    w_total: int,
    h_block: int,
    w_block: int,
) -> torch.Tensor:
    """Return 3-channel tensor [gray, p_long, p_short]."""
    if gray.dim() == 2:
        gray = gray.unsqueeze(0)
    _, h, w = gray.shape

    x0_block = (series - 1) * w_block
    x0_sub = x0_block + dx
    y0_sub = dy

    device = gray.device
    xs = torch.arange(w, device=device).float()
    ys = torch.arange(h, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    p_long = (x0_sub + xx + 0.5) / float(w_total)
    p_short = (y0_sub + yy + 0.5) / float(h_block)

    return torch.cat([gray, p_long.unsqueeze(0), p_short.unsqueeze(0)], dim=0)


def sincos_multi(*args, **kwargs):
    raise NotImplementedError("sincos_multi placeholder")
