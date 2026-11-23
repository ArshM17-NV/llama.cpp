#!/usr/bin/env python3
"""Convert a trained TreeFlash draft head and its target's token embeddings to a sidecar GGUF."""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "gguf-py"))
import gguf


@torch.inference_mode()
def convert(head: Path, target: Path, outfile: Path, device: str) -> None:
    checkpoint = torch.load(head, map_location="cpu", weights_only=True)
    state, cfg = checkpoint["state"], checkpoint["cfg"]
    if not (cfg.get("parent") and cfg.get("use_h") and cfg.get("mkfeat") and cfg.get("nl") == 1):
        raise ValueError("Expected a parent-conditioned TreeFlash head with one hidden layer and Markov features")
    depth, slots, candidates = state["T"].shape
    if (depth, slots, candidates) != (7, 21, 20):
        raise ValueError("Expected 7 draft positions and 20 candidates per position")

    reader = gguf.GGUFReader(target)
    emb = next((tensor for tensor in reader.tensors if tensor.name == "token_embd.weight"), None)
    if emb is None:
        raise ValueError(f"No token_embd.weight in {target}; pass the target GGUF containing its embeddings")
    width, vocab = map(int, emb.shape)
    if width != cfg["d_embed"]:
        raise ValueError(f"Target embedding width {width} differs from trained head width {cfg['d_embed']}")

    # The target GGUF may store embeddings in a quantized format. Use the exact table
    # that is served; projecting all vocabulary rows here avoids doing it at decode time.
    def project_weight(name):
        return state[name].to(device=device, dtype=torch.bfloat16)

    wv = project_weight("Wv.weight")
    wu = project_weight("Wu.weight")
    ws = project_weight("w.weight")
    vtable = np.empty((vocab, wv.shape[0]), dtype=np.float16)
    utable = np.empty_like(vtable)
    scalar = np.empty(vocab, dtype=np.float32)
    chunk_size = 256
    for start in range(0, vocab, chunk_size):
        stop = min(start + chunk_size, vocab)
        values = gguf.dequantize(emb.data[start:stop], emb.tensor_type)
        x = torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(device)
        x = (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)).bfloat16()
        vtable[start:stop] = F.linear(x, wv).half().cpu().numpy()
        u = F.linear(x, wu).float()
        utable[start:stop] = F.layer_norm(u, [u.shape[-1]]).half().cpu().numpy()
        scalar[start:stop] = F.linear(x, ws).float().flatten().cpu().numpy()

    writer = gguf.GGUFWriter(outfile, "treeflash")
    writer.add_string("treeflash.checkpoint_sha256", hashlib.sha256(head.read_bytes()).hexdigest())
    writer.add_string("treeflash.target_file", target.name)
    writer.add_tensor("embedding.value", vtable)
    writer.add_tensor("embedding.parent", utable)
    writer.add_tensor("embedding.bias", scalar)
    for key, tensor in state.items():
        if key in ("Wv.weight", "Wu.weight", "w.weight"):
            continue
        writer.add_tensor(key, tensor.float().contiguous().numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True, help="TreeFlash PyTorch checkpoint")
    parser.add_argument("--target", type=Path, required=True, help="Target model GGUF containing token_embd.weight")
    parser.add_argument("--outfile", type=Path, required=True, help="Destination GGUF sidecar")
    parser.add_argument("--device", default="cpu", help="Device used to precompute embedding projections, e.g. cuda")
    args = parser.parse_args()
    convert(args.head, args.target, args.outfile, args.device)
