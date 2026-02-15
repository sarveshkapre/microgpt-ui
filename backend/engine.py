"""Instrumented microGPT engine for training and visualization."""

from __future__ import annotations

import math
import os
import random
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
)


class Value:
    """A scalar value that tracks gradients through a computation graph."""

    def __init__(self, data: float, children: tuple["Value", ...] = (), local_grads: tuple[float, ...] = ()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Any) -> "Value":
        other = other if isinstance(other, Value) else Value(float(other))
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other: Any) -> "Value":
        other = other if isinstance(other, Value) else Value(float(other))
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float) -> "Value":
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self) -> "Value":
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self) -> "Value":
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self) -> "Value":
        return Value(max(0.0, self.data), (self,), (float(self.data > 0.0),))

    def __neg__(self) -> "Value":
        return self * -1.0

    def __radd__(self, other: Any) -> "Value":
        return self + other

    def __sub__(self, other: Any) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Any) -> "Value":
        return other + (-self)

    def __rmul__(self, other: Any) -> "Value":
        return self * other

    def __truediv__(self, other: Any) -> "Value":
        return self * (other**-1)

    def __rtruediv__(self, other: Any) -> "Value":
        return other * (self**-1)

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


@dataclass
class ModelConfig:
    n_embd: int = 16
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 8
    learning_rate: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    eps_adam: float = 1e-8
    num_steps: int = 500
    temperature: float = 0.5
    seed: int = 42


class MicroGPT:
    def __init__(self, cfg: ModelConfig, input_path: str = "input.txt"):
        self.cfg = cfg
        self.input_path = input_path
        self.rng = random.Random(cfg.seed)

        docs = self._load_docs()
        self.rng.shuffle(docs)
        self.docs = docs

        uchars = sorted(set("".join(self.docs)))
        self.BOS = len(uchars)
        self.itos = {i: ch for i, ch in enumerate(uchars)}
        self.stoi = {ch: i for i, ch in self.itos.items()}
        self.vocab_size = len(uchars) + 1

        self.head_dim = cfg.n_embd // cfg.n_head
        if self.head_dim * cfg.n_head != cfg.n_embd:
            raise ValueError("n_embd must be divisible by n_head")

        self.state_dict: dict[str, list[list[Value]]] = {}
        self._init_params()
        self.params = [p for mat in self.state_dict.values() for row in mat for p in row]
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        self.step = 0

    def _load_docs(self) -> list[str]:
        if not os.path.exists(self.input_path):
            urllib.request.urlretrieve(DEFAULT_DATA_URL, self.input_path)

        docs = [
            line.strip()
            for line in open(self.input_path, "r", encoding="utf-8").read().strip().split("\n")
            if line.strip()
        ]
        if not docs:
            raise ValueError(f"No documents found in {self.input_path}")
        return docs

    def _matrix(self, nout: int, nin: int, std: float = 0.02) -> list[list[Value]]:
        return [[Value(self.rng.gauss(0.0, std)) for _ in range(nin)] for _ in range(nout)]

    def _init_params(self) -> None:
        self.state_dict["wte"] = self._matrix(self.vocab_size, self.cfg.n_embd)
        self.state_dict["wpe"] = self._matrix(self.cfg.block_size, self.cfg.n_embd)
        self.state_dict["lm_head"] = self._matrix(self.vocab_size, self.cfg.n_embd)

        for i in range(self.cfg.n_layer):
            self.state_dict[f"layer{i}.attn_wq"] = self._matrix(self.cfg.n_embd, self.cfg.n_embd)
            self.state_dict[f"layer{i}.attn_wk"] = self._matrix(self.cfg.n_embd, self.cfg.n_embd)
            self.state_dict[f"layer{i}.attn_wv"] = self._matrix(self.cfg.n_embd, self.cfg.n_embd)
            self.state_dict[f"layer{i}.attn_wo"] = self._matrix(self.cfg.n_embd, self.cfg.n_embd, std=0.0)
            self.state_dict[f"layer{i}.mlp_fc1"] = self._matrix(4 * self.cfg.n_embd, self.cfg.n_embd)
            self.state_dict[f"layer{i}.mlp_fc2"] = self._matrix(self.cfg.n_embd, 4 * self.cfg.n_embd, std=0.0)

    def _linear(self, x: list[Value], w: list[list[Value]]) -> list[Value]:
        return [sum((wi * xi for wi, xi in zip(wo, x)), start=Value(0.0)) for wo in w]

    def _softmax(self, logits: list[Value]) -> list[Value]:
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps, start=Value(0.0))
        return [e / total for e in exps]

    def _rmsnorm(self, x: list[Value]) -> list[Value]:
        ms = sum((xi * xi for xi in x), start=Value(0.0)) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def _token_label(self, token_id: int) -> str:
        return "<BOS>" if token_id == self.BOS else self.itos[token_id]

    def _topk(self, probs: list[Value], k: int = 5) -> list[dict[str, Any]]:
        ranked = sorted(enumerate(probs), key=lambda x: x[1].data, reverse=True)[:k]
        return [
            {
                "token_id": idx,
                "token": self._token_label(idx),
                "prob": p.data,
            }
            for idx, p in ranked
        ]

    def _gpt(
        self,
        token_id: int,
        pos_id: int,
        keys: list[list[list[Value]]],
        values: list[list[list[Value]]],
        capture: bool = False,
    ) -> tuple[list[Value], list[dict[str, Any]]]:
        tok_emb = self.state_dict["wte"][token_id]
        pos_emb = self.state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = self._rmsnorm(x)

        attention_trace: list[dict[str, Any]] = []

        for li in range(self.cfg.n_layer):
            x_residual = x
            x = self._rmsnorm(x)
            q = self._linear(x, self.state_dict[f"layer{li}.attn_wq"])
            k = self._linear(x, self.state_dict[f"layer{li}.attn_wk"])
            v = self._linear(x, self.state_dict[f"layer{li}.attn_wv"])

            keys[li].append(k)
            values[li].append(v)

            x_attn: list[Value] = []
            layer_heads: list[dict[str, Any]] = []

            for h in range(self.cfg.n_head):
                hs = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in values[li]]

                attn_logits = [
                    sum((q_h[j] * k_h[t][j] for j in range(self.head_dim)), start=Value(0.0))
                    / (self.head_dim**0.5)
                    for t in range(len(k_h))
                ]
                attn_weights = self._softmax(attn_logits)
                head_out = [
                    sum((attn_weights[t] * v_h[t][j] for t in range(len(v_h))), start=Value(0.0))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)

                if capture:
                    layer_heads.append(
                        {
                            "head": h,
                            "weights": [w.data for w in attn_weights],
                            "logits": [l.data for l in attn_logits],
                        }
                    )

            x = self._linear(x_attn, self.state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = self._rmsnorm(x)
            x = self._linear(x, self.state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() ** 2 for xi in x]
            x = self._linear(x, self.state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

            if capture:
                attention_trace.append({"layer": li, "heads": layer_heads})

        logits = self._linear(x, self.state_dict["lm_head"])
        return logits, attention_trace

    def train_step(self) -> dict[str, Any]:
        cfg = self.cfg
        doc = self.docs[self.step % len(self.docs)]
        tokens = [self.BOS] + [self.stoi[ch] for ch in doc] + [self.BOS]
        n = min(cfg.block_size, len(tokens) - 1)
        if n <= 0:
            raise RuntimeError("Encountered empty token sequence")

        keys = [[] for _ in range(cfg.n_layer)]
        values = [[] for _ in range(cfg.n_layer)]
        losses: list[Value] = []
        token_events: list[dict[str, Any]] = []

        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]

            logits, attention_trace = self._gpt(token_id, pos_id, keys, values, capture=True)
            probs = self._softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

            token_events.append(
                {
                    "pos": pos_id,
                    "input_token_id": token_id,
                    "input_token": self._token_label(token_id),
                    "target_token_id": target_id,
                    "target_token": self._token_label(target_id),
                    "top_probs": self._topk(probs, 5),
                    "attention": attention_trace,
                }
            )

        loss = (1.0 / n) * sum(losses, start=Value(0.0))
        loss.backward()

        grad_norm = math.sqrt(sum(p.grad * p.grad for p in self.params))

        lr_t = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * self.step / cfg.num_steps))
        total_abs_update = 0.0

        for i, p in enumerate(self.params):
            self.m[i] = cfg.beta1 * self.m[i] + (1.0 - cfg.beta1) * p.grad
            self.v[i] = cfg.beta2 * self.v[i] + (1.0 - cfg.beta2) * p.grad**2
            m_hat = self.m[i] / (1.0 - cfg.beta1 ** (self.step + 1))
            v_hat = self.v[i] / (1.0 - cfg.beta2 ** (self.step + 1))
            delta = lr_t * m_hat / (v_hat**0.5 + cfg.eps_adam)
            p.data -= delta
            total_abs_update += abs(delta)
            p.grad = 0.0

        self.step += 1

        return {
            "type": "train_step",
            "step": self.step,
            "num_steps": cfg.num_steps,
            "doc": doc,
            "loss": loss.data,
            "learning_rate": lr_t,
            "grad_norm": grad_norm,
            "mean_abs_update": total_abs_update / max(1, len(self.params)),
            "sequence": {
                "token_ids": tokens[: n + 1],
                "tokens": [self._token_label(t) for t in tokens[: n + 1]],
            },
            "token_events": token_events,
            "last_token_event": token_events[-1],
        }

    def sample(self, num_samples: int = 5, temperature: float | None = None) -> list[str]:
        temperature = temperature if temperature is not None else self.cfg.temperature
        outputs: list[str] = []

        for _ in range(num_samples):
            keys = [[] for _ in range(self.cfg.n_layer)]
            values = [[] for _ in range(self.cfg.n_layer)]
            token_id = self.BOS
            sample_chars: list[str] = []

            for pos_id in range(self.cfg.block_size):
                logits, _ = self._gpt(token_id, pos_id, keys, values, capture=False)
                probs = self._softmax([l / temperature for l in logits])
                token_id = self.rng.choices(
                    range(self.vocab_size), weights=[p.data for p in probs], k=1
                )[0]
                if token_id == self.BOS:
                    break
                sample_chars.append(self.itos[token_id])

            outputs.append("".join(sample_chars))

        return outputs

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "num_docs": len(self.docs),
            "vocab_size": self.vocab_size,
            "bos_token_id": self.BOS,
            "step": self.step,
            "num_params": len(self.params),
        }

