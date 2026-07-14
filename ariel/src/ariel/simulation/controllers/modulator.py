"""Modulator: dynamic-features → gated residual on CPG parameters.

Per-joint, per-parameter modulator that takes a dynamic-feature vector
and produces a residual update to the CPG parameters set by the
``CPGDecoder``. Each timestep:

    output = base_params + gate * delta

- ``delta`` is a 2-hidden-layer MLP that proposes a *signed* perturbation
  to every CPG parameter of every joint. Its output is multiplied by
  ``delta_scales`` so the perturbation lives in CPG-parameter units.
- ``gate`` is a 1-hidden-layer MLP ending in sigmoid; one gate value per
  CPG parameter per joint. Lets evolution learn *which* parameters to
  touch and *when*.
- Combined as a residual on top of the decoder's static base params.
  When ``gate → 0`` everywhere the modulator is transparent and the
  controller behaves as decoder-only.

Notes
-----
- ``delta_scales`` should be a fraction (e.g. 10-20%) of the decoder's
  ``param_scales`` to keep the CPG limit cycle stable: the modulator
  perturbs the operating point but cannot rewrite the gait from scratch
  each timestep.
- ``gate_init_bias`` biases the gate network's final-layer bias toward
  a large negative value so ``sigmoid(.) ≈ 0`` at initialisation. This
  gives evolution a near-transparent starting point — the EA has to
  earn every degree of modulation.
- Dynamic features are typically per-joint (e.g. joint angle, joint
  velocity) optionally concatenated with global state (heading, roll,
  pitch) replicated across rows. The modulator does not need joint
  static features — joint identity is already captured by the decoder.
- ``set_flat_params`` / ``get_flat_params`` mirror ``CPGDecoder`` and
  ``NaCPG`` so the genome slice serialises identically. Layout:
  delta-network weights followed by gate-network weights.
"""

import math

import torch
from torch import nn

_DEFAULT_DELTA_SCALE = 0.1 * 2 * math.pi  # ≈ 10% of 2π, ~0.628
N_DYNAMIC_FEATURES = 16

class Modulator(nn.Module):
    """Gated residual modulator over CPG parameters.

    Parameters
    ----------
    dyn_dim : int
        Per-joint dynamic-feature dimensionality.
    out_dim : int, optional
        Number of CPG parameters per joint to modulate. Must match the
        decoder's ``out_dim``. Default is ``3``.
    hidden : int, optional
        Hidden-layer width for both delta and gate networks.
        Default is ``16``.
    delta_scales : tuple[float, ...] or None, optional
        Per-output multiplicative scale for the delta network's raw
        output. Length must equal ``out_dim``. ``None`` defaults to
        ``(0.628, 0.628, 0.628)`` ≈ 10% of ``2π`` — a conservative
        bound that keeps the CPG limit cycle stable. Provide explicit
        values to override.
    gate_init_bias : float, optional
        Constant added to the gate network's final-layer bias at
        initialisation, pre-sigmoid. ``-3.0`` gives ``sigmoid ≈ 0.047``
        — modulator starts near-transparent. Default is ``-3.0``.
    seed : int or None, optional
        If given, seeds ``torch`` before parameter initialisation for
        reproducibility. Default is ``None``.
    """

    def __init__(
        self,
        dyn_dim: int,
        out_dim: int = 3,
        hidden: int = 8,
        delta_scales: tuple[float, ...] | None = None,
        gate_init_bias: float = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if delta_scales is None:
            delta_scales = (_DEFAULT_DELTA_SCALE,) * out_dim
        if len(delta_scales) != out_dim:
            msg = (
                f"delta_scales length {len(delta_scales)} does not match "
                f"out_dim {out_dim}"
            )
            raise ValueError(msg)

        self.delta_net = nn.Sequential(
            nn.Linear(dyn_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(dyn_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )

        # Bias gate toward closed at init: sigmoid(-3) ≈ 0.047
        nn.init.constant_(self.gate_net[-2].bias, gate_init_bias)

        for p in self.delta_net.parameters():
            p.requires_grad = False
        for p in self.gate_net.parameters():
            p.requires_grad = False

        self.register_buffer(
            "delta_scales",
            torch.tensor(delta_scales, dtype=torch.float32),
        )

        self.dyn_dim = dyn_dim
        self.out_dim = out_dim

    @property
    def num_params(self) -> int:
        """Total number of evolved weights (genome-slice length)."""
        return sum(p.numel() for p in self.delta_net.parameters()) + sum(
            p.numel() for p in self.gate_net.parameters()
        )

    @property
    def num_delta_params(self) -> int:
        """Number of weights in the delta network."""
        return sum(p.numel() for p in self.delta_net.parameters())

    @property
    def num_gate_params(self) -> int:
        """Number of weights in the gate network."""
        return sum(p.numel() for p in self.gate_net.parameters())

    def delta_and_gate(
        self,
        dynamic_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (delta, gate) for inspection and logging.

        Parameters
        ----------
        dynamic_features : torch.Tensor
            Per-joint dynamic-feature tensor of shape ``[N, dyn_dim]``.

        Returns
        -------
        delta : torch.Tensor
            Scaled delta tensor of shape ``[N, out_dim]``.
        gate : torch.Tensor
            Gate tensor of shape ``[N, out_dim]``, in ``(0, 1)``.
        """
        with torch.inference_mode():
            delta = self.delta_net(dynamic_features) * self.delta_scales
            gate = self.gate_net(dynamic_features)
            return delta, gate

    def forward(
        self,
        dynamic_features: torch.Tensor,
        base_params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute modulated CPG parameters.

        Parameters
        ----------
        dynamic_features : torch.Tensor
            Per-joint dynamic-feature tensor of shape ``[N, dyn_dim]``.
        base_params : torch.Tensor
            Decoder output, the static CPG parameter operating point,
            shape ``[N, out_dim]``.

        Returns
        -------
        torch.Tensor
            Modulated CPG parameters of shape ``[N, out_dim]``,
            equal to ``base_params + gate * delta``.
        """
        with torch.inference_mode():
            delta = self.delta_net(dynamic_features) * self.delta_scales
            gate = self.gate_net(dynamic_features)
            return base_params + gate * delta

    def set_flat_params(self, params: torch.Tensor) -> None:
        """Load weights from a flat tensor.

        Layout: delta-network weights first, then gate-network weights.

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
        for p in self.delta_net.parameters():
            n = p.numel()
            p.data = params[pointer : pointer + n].view_as(p)
            pointer += n
        for p in self.gate_net.parameters():
            n = p.numel()
            p.data = params[pointer : pointer + n].view_as(p)
            pointer += n

    def get_flat_params(self) -> torch.Tensor:
        """Return all weights as a flat tensor (delta then gate)."""
        return torch.cat(
            [p.flatten() for p in self.delta_net.parameters()]
            + [p.flatten() for p in self.gate_net.parameters()],
        )

def get_num_params_mod():
    mod = modulator = Modulator(
        dyn_dim=N_DYNAMIC_FEATURES,
    )