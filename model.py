from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, Statevector


def _normalize_patch_size(patch_size: int | Sequence[int]) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return (patch_size, patch_size)
    if len(patch_size) != 2:
        raise ValueError("patch_size must be an int or length-2 sequence")
    return (int(patch_size[0]), int(patch_size[1]))


def split_patch_angles(image: torch.Tensor, patch_size: int | Sequence[int]) -> torch.Tensor:
    """
    Args:
        image: Tensor [C, H, W] with values in [0, 255].
    Returns:
        angles: Tensor [num_patches, C * patch_area] in [0, pi].
    """
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W]")
    patch_h, patch_w = _normalize_patch_size(patch_size)
    unfold = F.unfold(image.unsqueeze(0), kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    patches = unfold.squeeze(0).t()  # [num_patches, C * patch_area]
    angles = patches.clamp(0, 255).float() * (math.pi / 255.0)
    return angles


def encode_angles(circuit: QuantumCircuit, angles: Sequence[float], num_qubits: int = 8) -> None:
    idx = 0
    total = len(angles)
    while idx < total:
        for q in range(num_qubits):
            if idx >= total:
                break
            circuit.ry(float(angles[idx]), q)
            idx += 1
        for q in range(num_qubits):
            if idx >= total:
                break
            circuit.rx(float(angles[idx]), q)
            idx += 1


def add_vqc_layers(
    circuit: QuantumCircuit, params: ParameterVector, start: int, num_layers: int, num_qubits: int = 8
) -> int:
    """
    Adds layers and returns next free parameter index.
    """
    p = start
    for _ in range(num_layers):
        for q in range(num_qubits):
            circuit.cx(q, (q + 1) % num_qubits)
        for q in range(num_qubits):
            circuit.rx(params[p], q)
            p += 1
        for q in range(num_qubits):
            circuit.ry(params[p], q)
            p += 1
    return p


def _z_pauli(label_qubits: Iterable[int], num_qubits: int) -> Pauli:
    z = ["I"] * num_qubits
    for q in label_qubits:
        z[num_qubits - q - 1] = "Z"
    return Pauli("".join(z))


def _get_statevector(
    circuit: QuantumCircuit, param_binds: Sequence[float], backend=None
) -> Statevector:
    bound = circuit.assign_parameters(param_binds, inplace=False)
    if backend is None:
        return Statevector.from_instruction(bound)
    job = backend.run(bound)
    result = job.result()
    data = np.array(result.get_statevector(bound), dtype=complex)
    return Statevector(data)


def statevector_features(circuit: QuantumCircuit, param_binds: Sequence[float], backend=None) -> np.ndarray:
    sv = _get_statevector(circuit, param_binds, backend)
    data = sv.data
    return np.concatenate([data.real, data.imag])


def correlation_features(circuit: QuantumCircuit, param_binds: Sequence[float], backend=None) -> np.ndarray:
    sv = _get_statevector(circuit, param_binds, backend)
    n = circuit.num_qubits
    values: List[float] = []

    singles = [_z_pauli([i], n) for i in range(n)]
    pairs = [_z_pauli(c, n) for c in combinations(range(n), 2)]
    triples = [_z_pauli(c, n) for c in combinations(range(n), 3)]

    for op in singles + pairs + triples:
        values.append(float(np.real(sv.expectation_value(op))))
    return np.array(values, dtype=np.float64)


def _comb(n: int, r: int) -> int:
    return math.comb(n, r) if n >= r else 0


def measurement_dim(num_qubits: int, measurement: str) -> int:
    measurement = measurement.lower()
    if measurement == "statevector":
        return 2 ** (num_qubits + 1)
    if measurement == "correlations":
        n = num_qubits
        return n + _comb(n, 2) + _comb(n, 3)
    raise ValueError("measurement must be 'statevector' or 'correlations'")


@dataclass
class QuantumAnsatz:
    num_qubits: int = 8
    vqc_layers: int = 1
    measurement: str = "statevector"
    backend_device: str = "cpu"  # "cpu" or "gpu"

    def __post_init__(self) -> None:
        self.measurement = self.measurement.lower()
        self.params = ParameterVector("theta", self.vqc_layers * 2 * self.num_qubits)
        self.backend = None
        if self.backend_device.lower() == "gpu":
            try:
                from qiskit_aer import AerSimulator
            except ImportError as exc:
                raise ImportError("qiskit-aer-gpu is required for GPU backend") from exc
            self.backend = AerSimulator(method="statevector", device="GPU")

    @property
    def param_shape(self) -> int:
        return len(self.params)

    @property
    def feature_dim(self) -> int:
        return measurement_dim(self.num_qubits, self.measurement)

    def circuit_for_angles(self, angles: Sequence[float]) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        encode_angles(circuit, angles, num_qubits=self.num_qubits)
        add_vqc_layers(circuit, self.params, 0, self.vqc_layers, num_qubits=self.num_qubits)
        return circuit

    def features(self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None) -> np.ndarray:
        if param_values is None:
            param_values = [0.0] * self.param_shape
        if len(param_values) != self.param_shape:
            raise ValueError(f"param_values must have length {self.param_shape}")
        circuit = self.circuit_for_angles(patch_angles)
        if self.measurement == "statevector":
            return statevector_features(circuit, param_values, self.backend)
        if self.measurement == "correlations":
            return correlation_features(circuit, param_values, self.backend)
        raise ValueError("measurement must be 'statevector' or 'correlations'")


@dataclass
class QuantumPatchModel:
    patch_size: int | Sequence[int] = 4
    num_qubits: int = 8
    vqc_layers: int = 1
    measurement: str = "statevector"  # "statevector" or "correlations"

    def __post_init__(self) -> None:
        self.patch_size = _normalize_patch_size(self.patch_size)
        self.ansatz = QuantumAnsatz(self.num_qubits, self.vqc_layers, self.measurement)
        self.params = self.ansatz.params

    @property
    def param_shape(self) -> int:
        return self.ansatz.param_shape

    def circuit_for_angles(self, angles: Sequence[float]) -> QuantumCircuit:
        return self.ansatz.circuit_for_angles(angles)

    def features(self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None) -> np.ndarray:
        return self.ansatz.features(patch_angles, param_values)

    def image_patch_features(
        self, image: torch.Tensor, param_values: Sequence[float] | None = None
    ) -> List[np.ndarray]:
        angles = split_patch_angles(image, self.patch_size)
        return [self.features(a.tolist(), param_values) for a in angles]


@dataclass
class SeparateQKV:
    query_ansatz: QuantumAnsatz
    key_ansatz: QuantumAnsatz
    value_ansatz: QuantumAnsatz

    def qkv_from_patch(
        self,
        patch_angles: Sequence[float],
        params_q: Sequence[float] | None = None,
        params_k: Sequence[float] | None = None,
        params_v: Sequence[float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.query_ansatz.features(patch_angles, params_q)
        k = self.key_ansatz.features(patch_angles, params_k)
        v = self.value_ansatz.features(patch_angles, params_v)
        return q, k, v

    def qkv_from_image(
        self,
        image: torch.Tensor,
        patch_size: int | Sequence[int],
        params_q: Sequence[float] | None = None,
        params_k: Sequence[float] | None = None,
        params_v: Sequence[float] | None = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        angles = split_patch_angles(image, patch_size)
        return [self.qkv_from_patch(a.tolist(), params_q, params_k, params_v) for a in angles]


class SharedQKVProjector(nn.Module):
    def __init__(
        self,
        ansatz: QuantumAnsatz,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.ansatz = ansatz
        self.device = torch.device(device) if device is not None else None
        fdim = ansatz.feature_dim
        self.query = nn.Linear(fdim, q_dim, bias=False)
        self.key = nn.Linear(fdim, k_dim, bias=False)
        self.value = nn.Linear(fdim, v_dim, bias=False)

    def forward_patch(
        self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.ansatz.features(patch_angles, param_values)
        x = torch.as_tensor(feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return (
            self.query(x).squeeze(0),
            self.key(x).squeeze(0),
            self.value(x).squeeze(0),
        )

    def forward_image(
        self, image: torch.Tensor, patch_size: int | Sequence[int], param_values: Sequence[float] | None = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        angles = split_patch_angles(image, patch_size)
        return [self.forward_patch(a.tolist(), param_values) for a in angles]


def _stack_list(features: Sequence[np.ndarray] | Sequence[torch.Tensor], device: torch.device | None = None) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("features is empty")
    if isinstance(features[0], torch.Tensor):
        return torch.stack([f.to(device=device) for f in features], dim=0)
    return torch.stack([torch.from_numpy(np.asarray(f)).to(device=device) for f in features], dim=0)


class AttentionLayer(nn.Module):
    def __init__(self, attn_type: str = "dot", gamma: float = 1.0) -> None:
        super().__init__()
        self.attn_type = attn_type.lower()
        self.gamma = gamma

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: [B, P, D]
        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        if self.attn_type == "dot":
            scale = q.size(-1) ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
        elif self.attn_type == "rbf":
            # squared euclidean distances
            q_exp = q.unsqueeze(-2)  # [B, P, 1, D]
            k_exp = k.unsqueeze(-3)  # [B, 1, P, D]
            dist2 = (q_exp - k_exp).pow(2).sum(-1)
            weights = torch.softmax(-self.gamma * dist2, dim=-1)
        else:
            raise ValueError("attn_type must be 'dot' or 'rbf'")
        return torch.matmul(weights, v)


class StackedSelfAttention(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1, attn_type: str = "dot", gamma: float = 1.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionLayer(attn_type, gamma) for _ in range(num_layers)])
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_layers)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: [B, P, D] or [P, D]
        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        x = v
        for attn, proj in zip(self.layers, self.proj):
            out = attn(q, k, x)
            x = proj(out)
        return x


def aggregate_patches(patch_feats: torch.Tensor, mode: str = "concat") -> torch.Tensor:
    """
    Args:
        patch_feats: [B, P, D] or [P, D]
        mode: "concat" or "gap_gmp"
    Returns:
        embedding: [B, P*D] for concat, or [B, 2*D] for gap_gmp
    """
    if patch_feats.dim() == 2:
        patch_feats = patch_feats.unsqueeze(0)
    if mode == "concat":
        return patch_feats.flatten(start_dim=1)
    if mode == "gap_gmp":
        gap = patch_feats.mean(dim=1)
        gmp = patch_feats.amax(dim=1)
        return torch.cat([gap, gmp], dim=-1)
    raise ValueError("mode must be 'concat' or 'gap_gmp'")


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int] | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim] + (list(hidden_dims) if hidden_dims else [])
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)
