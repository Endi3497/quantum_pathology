from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


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


def encode_params(circuit: QuantumCircuit, params: Sequence, num_qubits: int = 8) -> None:
    idx = 0
    total = len(params)
    while idx < total:
        for q in range(num_qubits):
            if idx >= total:
                break
            circuit.ry(params[idx], q)
            idx += 1
        for q in range(num_qubits):
            if idx >= total:
                break
            circuit.rx(params[idx], q)
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
    from qiskit.quantum_info import Pauli

    z = ["I"] * num_qubits
    for q in label_qubits:
        z[num_qubits - q - 1] = "Z"
    return Pauli("".join(z))


def _get_statevector(circuit, param_bind: dict, backend=None):
    from qiskit.quantum_info import Statevector

    if backend is None:
        bound = circuit.assign_parameters(param_bind, inplace=False)
        return Statevector.from_instruction(bound)
    job = backend.run(circuit, parameter_binds=[param_bind])
    result = job.result()
    data = np.array(result.get_statevector(circuit, param_bind), dtype=complex)
    return Statevector(data)


def statevector_features(circuit, param_bind: dict, backend=None) -> np.ndarray:
    sv = _get_statevector(circuit, param_bind, backend)
    data = sv.data
    return np.concatenate([data.real, data.imag])


def correlation_features(circuit, param_bind: dict, backend=None) -> np.ndarray:
    sv = _get_statevector(circuit, param_bind, backend)
    n = circuit.num_qubits
    values: List[float] = []

    singles = [_z_pauli([i], n) for i in range(n)]
    pairs = [_z_pauli(c, n) for c in combinations(range(n), 2)]
    triples = [_z_pauli(c, n) for c in combinations(range(n), 3)]

    for op in singles + pairs + triples:
        values.append(float(np.real(sv.expectation_value(op))))
    return np.array(values, dtype=np.float64)


def _apply_single_qubit(state: torch.Tensor, gate: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    """
    state: [B, 2**n]
    gate: [2, 2] (shared) or [B, 2, 2] (per batch)
    """
    bsz = state.shape[0]
    state = state.reshape(bsz, *([2] * num_qubits))
    state = state.movedim(qubit + 1, -1)
    state = state.reshape(bsz, -1, 2)

    if gate.dim() == 2:
        gate = gate.unsqueeze(0)
    if gate.dim() == 3 and gate.shape[0] != bsz and gate.shape[-1] == bsz:
        gate = gate.permute(2, 0, 1)
    if gate.dim() == 3 and gate.shape[0] == 1:
        gate = gate.expand(bsz, -1, -1)

    # state [B, N, 2], gate [B, 2, 2]
    state = torch.matmul(state, gate.transpose(1, 2))

    state = state.reshape(bsz, *([2] * (num_qubits - 1)), 2)
    state = state.movedim(-1, qubit + 1)
    return state.reshape(bsz, -1).contiguous()


def _apply_cnot(state: torch.Tensor, control: int, target: int, num_qubits: int) -> torch.Tensor:
    if control == target:
        return state
    bsz = state.shape[0]
    state = state.reshape(bsz, *([2] * num_qubits))
    state = state.movedim([control + 1, target + 1], [-2, -1])
    orig_shape = state.shape
    state = state.reshape(-1, 4)
    cnot = state.new_tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=state.dtype
    )
    state = torch.matmul(state, cnot.t())
    state = state.reshape(orig_shape)
    state = state.movedim([-2, -1], [control + 1, target + 1])
    return state.reshape(bsz, -1).contiguous()


def simulate_torch_features(
    angles: torch.Tensor,
    theta: torch.Tensor,
    num_qubits: int,
    vqc_layers: int,
    measurement: str,
) -> torch.Tensor:
    """
    Differentiable torch simulation (statevector) with batch support.
    angles: [B, data_dim] or [data_dim]
    theta: [param_shape]
    returns: [B, feature_dim] or [feature_dim]
    """
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)
    if theta.dim() != 1:
        raise ValueError("theta must be 1-D")
    batch = angles.shape[0]

    device = theta.device
    dtype = torch.complex64 if theta.dtype == torch.float32 else torch.complex128
    angles = angles.to(device=device, dtype=theta.dtype)

    state = torch.zeros(batch, 2**num_qubits, device=device, dtype=dtype)
    state[:, 0] = 1.0 + 0j

    def rx_gate(phi):
        if phi.dim() == 0:
            c = torch.cos(phi / 2)
            s = torch.sin(phi / 2)
            gate = torch.stack(
                [
                    torch.stack([c, -1j * s]),
                    torch.stack([-1j * s, c]),
                ]
            )
        else:
            c = torch.cos(phi / 2)
            s = torch.sin(phi / 2)
            gate = torch.zeros(phi.shape[0], 2, 2, device=device, dtype=dtype)
            gate[:, 0, 0] = c
            gate[:, 0, 1] = -1j * s
            gate[:, 1, 0] = -1j * s
            gate[:, 1, 1] = c
        return gate

    def ry_gate(phi):
        if phi.dim() == 0:
            c = torch.cos(phi / 2)
            s = torch.sin(phi / 2)
            gate = torch.stack(
                [
                    torch.stack([c, -s]),
                    torch.stack([s, c]),
                ]
            )
        else:
            c = torch.cos(phi / 2)
            s = torch.sin(phi / 2)
            gate = torch.zeros(phi.shape[0], 2, 2, device=device, dtype=dtype)
            gate[:, 0, 0] = c
            gate[:, 0, 1] = -s
            gate[:, 1, 0] = s
            gate[:, 1, 1] = c
        return gate

    idx = 0
    total = angles.shape[1]
    for _ in range(math.ceil(total / num_qubits / 2)):
        for q in range(num_qubits):
            if idx >= total:
                break
            gate = ry_gate(angles[:, idx]).to(device=device, dtype=dtype)
            state = _apply_single_qubit(state, gate, q, num_qubits)
            idx += 1
        for q in range(num_qubits):
            if idx >= total:
                break
            gate = rx_gate(angles[:, idx]).to(device=device, dtype=dtype)
            state = _apply_single_qubit(state, gate, q, num_qubits)
            idx += 1

    idx_theta = 0
    for _ in range(vqc_layers):
        for q in range(num_qubits):
            state = _apply_cnot(state, q, (q + 1) % num_qubits, num_qubits)
        for q in range(num_qubits):
            gate = rx_gate(theta[idx_theta]).to(device=device, dtype=dtype)
            state = _apply_single_qubit(state, gate, q, num_qubits)
            idx_theta += 1
        for q in range(num_qubits):
            gate = ry_gate(theta[idx_theta]).to(device=device, dtype=dtype)
            state = _apply_single_qubit(state, gate, q, num_qubits)
            idx_theta += 1

    if measurement == "statevector":
        out = torch.cat([state.real, state.imag], dim=-1)
    elif measurement == "correlations":
        probs = state.abs() ** 2  # [B, 2^n]
        idxs = torch.arange(probs.shape[1], device=device)
        values = []
        for bits in ([i] for i in range(num_qubits)):
            parity = ((idxs.unsqueeze(1) >> torch.tensor(bits, device=device)) & 1).sum(dim=1) % 2
            eig = 1 - 2 * parity
            values.append((probs * eig).sum(dim=1))
        from itertools import combinations

        for r in [2, 3]:
            for combo in combinations(range(num_qubits), r):
                parity = ((idxs.unsqueeze(1) >> torch.tensor(combo, device=device)) & 1).sum(dim=1) % 2
                eig = 1 - 2 * parity
                values.append((probs * eig).sum(dim=1))
        out = torch.stack(values, dim=1)
    else:
        raise ValueError("measurement must be 'statevector' or 'correlations'")

    if out.shape[0] == 1:
        return out.squeeze(0)
    return out


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
    data_dim: int
    num_qubits: int = 8
    vqc_layers: int = 1
    measurement: str = "statevector"
    backend_device: str = "cpu"  # "cpu" or "gpu"
    use_torch_autograd: bool = True

    def __post_init__(self) -> None:
        self.measurement = self.measurement.lower()
        self.params = None
        self.data_params = None
        self.backend = None
        self.template = None
        if not self.use_torch_autograd:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector

            self.params = ParameterVector("theta", self.param_shape)
            self.data_params = ParameterVector("x", self.data_dim)
            if self.backend_device.lower() == "gpu":
                try:
                    from qiskit_aer import AerSimulator
                except ImportError as exc:
                    raise ImportError("qiskit-aer-gpu is required for GPU backend") from exc
                self.backend = AerSimulator(method="statevector", device="GPU")
            self.template = self._build_template()

    @property
    def param_shape(self) -> int:
        return self.vqc_layers * 2 * self.num_qubits

    @property
    def feature_dim(self) -> int:
        return measurement_dim(self.num_qubits, self.measurement)

    def _build_template(self) -> QuantumCircuit:
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(self.num_qubits)
        encode_params(circuit, self.data_params, num_qubits=self.num_qubits)
        add_vqc_layers(circuit, self.params, 0, self.vqc_layers, num_qubits=self.num_qubits)
        return circuit

    def circuit_for_angles(
        self, angles: Sequence[float], param_values: Sequence[float] | None = None
    ) -> QuantumCircuit:
        if param_values is None:
            param_values = [0.0] * self.param_shape
        if len(param_values) != self.param_shape:
            raise ValueError(f"param_values must have length {self.param_shape}")
        if len(angles) != self.data_dim:
            raise ValueError(f"angles length {len(angles)} does not match data_dim {self.data_dim}")
        bind = {self.data_params[i]: float(angles[i]) for i in range(self.data_dim)}
        bind.update({self.params[i]: float(param_values[i]) for i in range(self.param_shape)})
        return self.template.assign_parameters(bind, inplace=False)

    def features(self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None) -> np.ndarray:
        if self.use_torch_autograd:
            angles_t = torch.as_tensor(patch_angles, dtype=torch.float32)
            theta_t = torch.as_tensor(
                param_values if param_values is not None else [0.0] * self.param_shape, dtype=torch.float32
            )
            return self.torch_features(angles_t, theta_t).detach().cpu().numpy()
        if param_values is None:
            param_values = [0.0] * self.param_shape
        if len(param_values) != self.param_shape:
            raise ValueError(f"param_values must have length {self.param_shape}")
        if len(patch_angles) != self.data_dim:
            raise ValueError(f"patch_angles length {len(patch_angles)} does not match data_dim {self.data_dim}")
        bind = {self.data_params[i]: float(patch_angles[i]) for i in range(self.data_dim)}
        bind.update({self.params[i]: float(param_values[i]) for i in range(self.param_shape)})
        if self.measurement == "statevector":
            return statevector_features(self.template, bind, self.backend)
        if self.measurement == "correlations":
            return correlation_features(self.template, bind, self.backend)
        raise ValueError("measurement must be 'statevector' or 'correlations'")

    def torch_features(self, patch_angles: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if not self.use_torch_autograd:
            # Fallback: use numpy path (non-differentiable)
            feats = self.features(
                patch_angles.detach().cpu().numpy().tolist(),
                theta.detach().cpu().numpy().tolist(),
            )
            return torch.as_tensor(feats, dtype=theta.dtype, device=theta.device)
        return simulate_torch_features(
            patch_angles,
            theta,
            num_qubits=self.num_qubits,
            vqc_layers=self.vqc_layers,
            measurement=self.measurement,
        )


@dataclass
class QuantumPatchModel:
    patch_size: int | Sequence[int] = 4
    channels: int = 2
    num_qubits: int = 8
    vqc_layers: int = 1
    measurement: str = "statevector"  # "statevector" or "correlations"
    backend_device: str = "cpu"
    use_torch_autograd: bool = True

    def __post_init__(self) -> None:
        self.patch_size = _normalize_patch_size(self.patch_size)
        patch_h, patch_w = self.patch_size
        data_dim = self.channels * patch_h * patch_w
        self.ansatz = QuantumAnsatz(
            data_dim=data_dim,
            num_qubits=self.num_qubits,
            vqc_layers=self.vqc_layers,
            measurement=self.measurement,
            backend_device=self.backend_device,
            use_torch_autograd=self.use_torch_autograd,
        )
        self.params = self.ansatz.params

    @property
    def param_shape(self) -> int:
        return self.ansatz.param_shape

    def circuit_for_angles(self, angles: Sequence[float], param_values: Sequence[float] | None = None) -> QuantumCircuit:
        return self.ansatz.circuit_for_angles(angles, param_values)

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
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.ansatz = ansatz
        self.device = torch.device(device) if device is not None else None
        self.trainable = trainable
        if trainable:
            self.theta = nn.Parameter(torch.zeros(ansatz.param_shape, dtype=torch.float32))
        fdim = ansatz.feature_dim
        self.query = nn.Linear(fdim, q_dim, bias=False)
        self.key = nn.Linear(fdim, k_dim, bias=False)
        self.value = nn.Linear(fdim, v_dim, bias=False)

    def forward_patch(
        self, patch_angles: Sequence[float], param_values: Sequence[float] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.trainable:
            angles_t = torch.as_tensor(patch_angles, dtype=torch.float32, device=self.device)
            x = self.ansatz.torch_features(angles_t, self.theta).unsqueeze(0)
        else:
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
        if self.trainable:
            angle_tensor = torch.stack([a for a in angles], dim=0).to(self.device or angles.device)
            feats = self.ansatz.torch_features(angle_tensor, self.theta)
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)
            q = self.query(feats)
            k = self.key(feats)
            v = self.value(feats)
            return [(q[i], k[i], v[i]) for i in range(q.shape[0])]
        return [self.forward_patch(a.tolist(), param_values) for a in angles]


def _stack_list(features: Sequence[np.ndarray] | Sequence[torch.Tensor], device: torch.device | None = None) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("features is empty")
    if isinstance(features[0], torch.Tensor):
        return torch.stack([f.to(device=device) for f in features], dim=0)
    return torch.stack([torch.from_numpy(np.asarray(f)).to(device=device) for f in features], dim=0)


class HybridQuantumClassifier(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int | Sequence[int],
        ansatz: QuantumAnsatz,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        attn_layers: int,
        attn_type: str,
        rbf_gamma: float,
        agg_mode: str,
        hidden_dims: Sequence[int],
        dropout: float,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.patch_size = _normalize_patch_size(patch_size)
        self.patch_count = (image_size // self.patch_size[0]) * (image_size // self.patch_size[1])
        self.qkv = SharedQKVProjector(
            ansatz,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            device=self.device,
            trainable=True,
        )
        self.attn = StackedSelfAttention(dim=v_dim, num_layers=attn_layers, attn_type=attn_type, gamma=rbf_gamma)
        self.agg_mode = agg_mode
        in_dim = v_dim * self.patch_count if agg_mode == "concat" else v_dim * 2
        self.classifier = BinaryClassifier(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.to(self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError("images must be [B, C, H, W]")
        outputs = []
        for img in images:
            qkv_list = self.qkv.forward_image(img, self.patch_size)
            q = torch.stack([t[0] for t in qkv_list], dim=0).to(self.device)
            k = torch.stack([t[1] for t in qkv_list], dim=0).to(self.device)
            v = torch.stack([t[2] for t in qkv_list], dim=0).to(self.device)
            attn_out = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))  # [1, P, v_dim]
            emb = aggregate_patches(attn_out, mode=self.agg_mode)  # [1, in_dim]
            outputs.append(emb.squeeze(0))
        feats = torch.stack(outputs, dim=0)
        return self.classifier(feats)


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
