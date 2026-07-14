"""CPGDecoder: per-joint static-feature → CPG-parameter MLP.

Maps a fixed-size per-joint feature vector to a fixed-size CPG-parameter
vector. The decoder weights form a fixed-length genome slice; the number
of joints in the robot is irrelevant to the genome size because the MLP
is applied independently to each joint's feature row.

Output channels (default ``out_dim=5``) align with ``NaCPG`` parameter
groups in this order: ``phase, amplitudes, w, ha, b``. Each channel can
optionally be scaled by ``param_scales`` so the raw MLP output is mapped
into the natural range of each parameter group.

Notes
-----
- Weights are registered as ``nn.Parameter`` with ``requires_grad=False``
  so they are evolved, not back-propagated.
- ``set_flat_params`` / ``get_flat_params`` mirror ``NaCPG`` so an EA
  operator can serialise the decoder weights as a flat tensor genome
  slice.
"""

import torch
from torch import nn
import math 
CPG_PARAM_SCALE = 2 * math.pi   # matches NaCPG init range for phase/amp/w
N_STATIC_FEATURES = 13


class CPGDecoder(nn.Module):
    """MLP that maps per-joint static features to CPG parameters.

    Parameters
    ----------
    in_dim : int
        Per-joint static-feature dimensionality.
    hidden : int, optional
        Number of units in each hidden layer. Default is ``16``.
    out_dim : int, optional
        Output dimensionality per joint. Default is ``5``, matching the
        ``NaCPG`` parameter groups ``(phase, amplitudes, w, ha, b)``.
    param_scales : tuple[float, ...] or None, optional
        Per-output multiplicative scale applied after the final linear
        layer. Length must equal ``out_dim``. ``None`` disables scaling
        (raw MLP output). Default is ``None``.
    seed : int or None, optional
        If given, seeds ``torch`` before parameter initialisation for
        reproducibility. Default is ``None``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 8,
        out_dim: int = 3,
        param_scales: tuple[float, ...] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if param_scales is not None and len(param_scales) != out_dim:
            msg = (
                f"param_scales length {len(param_scales)} does not match "
                f"out_dim {out_dim}"
            )
            raise ValueError(msg)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

        for p in self.net.parameters():
            p.requires_grad = False

        if param_scales is None:
            self.register_buffer("param_scales", torch.ones(out_dim))
        else:
            self.register_buffer(
                "param_scales",
                torch.tensor(param_scales, dtype=torch.float32),
            )

        self.in_dim = in_dim
        self.out_dim = out_dim

    @property
    def num_params(self) -> int:
        """Total number of evolved weights (genome-slice length)."""
        return sum(p.numel() for p in self.net.parameters())

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """Compute CPG parameters for every joint.

        Parameters
        ----------
        joint_features : torch.Tensor
            Per-joint feature tensor of shape ``[N, in_dim]``.

        Returns
        -------
        torch.Tensor
            CPG-parameter tensor of shape ``[N, out_dim]``, with each
            output channel multiplied by its entry in ``param_scales``.
        """
        with torch.inference_mode():
            out = self.net(joint_features)
            return out * self.param_scales

    def set_flat_params(self, params: torch.Tensor) -> None:
        """Load weights from a flat tensor.

        Parameters
        ----------
        params : torch.Tensor
            Flat parameter tensor of length ``num_params``.
        """
        if params.numel() != self.num_params:
            msg = (
                f"Parameter vector has incorrect size. "
                f"Expected {self.num_params}, got {params.numel()}."
            )
            raise ValueError(msg)

        pointer = 0
        for p in self.net.parameters():
            n = p.numel()
            p.data = params[pointer : pointer + n].view_as(p)
            pointer += n

    def get_flat_params(self) -> torch.Tensor:
        """Return all weights as a flat tensor."""
        return torch.cat([p.flatten() for p in self.net.parameters()])
    
def get_num_params_cpg() -> int:
    controller = CPGDecoder(
        in_dim=N_STATIC_FEATURES,
        out_dim=3,
        param_scales=(CPG_PARAM_SCALE,) * 3,
    )
    return sum(p.numel() for p in controller.parameters())
