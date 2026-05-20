# Review: Grace-Blackwell Unified Memory Optimization Patch

**Date:** 2026-05-20
**Reviewed files:**
- `patches/model_management.py` (1903 lines, patched)
- `docker-compose.yml` (environment variables, volume mount)

**Scope:** The patch modifies 3 locations in ComfyUI's `model_management.py`:
1. Unified memory detection + HIGH_VRAM override (new, lines 462–515)
2. `maximum_vram_for_weights()` — 95% instead of 88% (lines 968–974)
3. `soft_empty_cache()` — skip `empty_cache()` on unified memory (lines 1847–1854)

The patched file is a direct descendant of upstream ComfyUI (identical line count: 1903), with only these 3 changes applied. The docker-compose mounts it as a read-only volume override.

---

## CRITICAL

### C1. MPS `vram_state` override — SHARED silently replaced by HIGH_VRAM

**File:** `patches/model_management.py`, lines 460 vs 506–512

**Problem:** The execution order is:
```
line 460:  if cpu_state == CPUState.MPS: vram_state = VRAMState.SHARED
line 475:  _is_unified_memory() returns True for MPS
line 504:  UNIFIED_MEMORY = True
line 512:  vram_state = VRAMState.HIGH_VRAM          ← SHARED overwritten!
```

MPS (Apple Silicon) already has its own unified memory path (`VRAMState.SHARED`, documented as "No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both"). The patch overrides this with `HIGH_VRAM`, which has different semantics.

**Impact in this project:** None. MPS is unreachable in the Sparky Docker container (ARM64 Linux + NVIDIA CUDA).

**Impact if upstreamed:** This would break Apple Silicon behavior. `SHARED` and `HIGH_VRAM` are distinct states — for example, `unet_offload_device()` returns GPU for `HIGH_VRAM` but CPU for `SHARED` (line 942).

**Fix (2 options):**
- **Option A (preferred):** Remove MPS from `_is_unified_memory()`:
  ```python
  if cpu_state == CPUState.MPS:
      return False  # MPS already handles unified memory via VRAMState.SHARED
  ```
- **Option B:** Guard the override:
  ```python
  if UNIFIED_MEMORY and cpu_state != CPUState.MPS:
      vram_state = VRAMState.HIGH_VRAM
  ```

### C2. MPS `soft_empty_cache()` — `torch.mps.empty_cache()` silently skipped

**File:** `patches/model_management.py`, lines 1845–1858

**Problem:** On MPS, the execution flow is:
```
line 1846:  cpu_mode() → False (MPS ≠ CPU)
line 1851:  UNIFIED_MEMORY and not force → True (MPS detected as unified)
line 1854:  return          ← EXITS HERE
line 1857:  cpu_state == CPUState.MPS → NEVER REACHED
line 1858:  torch.mps.empty_cache() → NEVER CALLED
```

The unified memory early return at line 1854 sits **before** the MPS branch at line 1857, so `torch.mps.empty_cache()` is dead code when `_is_unified_memory()` returns True for MPS.

**Impact in this project:** None. MPS unreachable in CUDA container.

**Fix:** Same as C1. Removing MPS from `_is_unified_memory()` fixes both issues. Alternatively, move the MPS check before the unified memory check in `soft_empty_cache()`:
```python
if cpu_state == CPUState.MPS:
    torch.mps.empty_cache()
    return
if UNIFIED_MEMORY and not force:
    ...
```

---

## WARNING

### W1. Ratio-based detection false positive: VRAM > RAM on high-end discrete GPUs

**Scenario:** A system with GPU VRAM exceeding system RAM, e.g.:
- A100 80GB + 64GB system RAM → ratio = 1.25 > 0.95
- H100 80GB + 64GB system RAM → ratio = 1.25 > 0.95

The patch would classify these as unified memory and set HIGH_VRAM. However, in this case HIGH_VRAM is **actually desirable** — with more VRAM than RAM, you never want CPU offloading because models won't fit in system RAM. So this false positive has benign consequences.

**False positive that would be harmful:** A system where VRAM ≈ RAM but they are **physically separate** pools. Example: 48GB GPU + 48GB system RAM = ratio 1.0. With 48GB discrete VRAM, if a model is 46GB, you'd want CPU offloading for other models. The patch would prevent that. However, such configurations are essentially nonexistent in practice — discrete GPU VRAM is almost always substantially smaller than system RAM.

**Mitigation:** The device name string check (`'grace'`, `'gb10'`, `'gb200'`) provides an explicit detection path that would catch true Grace-Blackwell even if the ratio check missed.

**Risk:** Low. True false positives (harmful ones) require VRAM ≈ RAM on a non-unified system, which is rare.

### W2. `maximum_vram_for_weights()` reserves only 2GB — borderline for large batch inference

**File:** `patches/model_management.py`, lines 968–974

```python
if UNIFIED_MEMORY:
    return (get_total_memory(device) * 0.95 - 2 * 1024 * 1024 * 1024)
```

On the 128GB Spark, this yields ~119.6GB available for weights. The remaining ~8.4GB (128 × 0.05 + 2GB) covers:
- OS memory (kernel, system services)
- Docker container overhead
- PyTorch runtime (autocast buffers, CUDA contexts)
- Intermediate tensors during inference (activations, attention maps)
- VAE decode buffers

For typical ComfyUI workflows (SDXL, Flux, LTX-2), 8.4GB of headroom is generous. However, for extreme batch sizes or high-resolution video models that allocate large intermediate tensors, this could be tight.

**Comparison to upstream:** The upstream formula is `total * 0.88 - minimum_inference_memory()` where `minimum_inference_memory()` = 0.8GB + `extra_reserved_memory()` (0.4GB default) = ~1.2GB. On a discrete 24GB card: 24*0.88 - 1.2 = ~19.9GB. The patched formula on 128GB: 128*0.95 - 2 = ~119.6GB. The patch uses a larger fraction (95% vs 88%) which is appropriate for unified memory — there's no separate "VRAM pool" to protect.

**Recommendation:** Consider making the 2GB reserve configurable, or basing it on a fraction of total memory rather than a fixed value. 2GB is 1.5% of 128GB, which is reasonable, but if someone ran this on a hypothetical 32GB Grace-Blackwell system, 2GB would be 6.25%.

### W3. `'grace'` substring in device name is broad

**File:** `patches/model_management.py`, line 493

```python
is_gb = 'gb10' in device_name or 'gb200' in device_name or 'grace' in device_name
```

The substring `'grace'` could theoretically match future unrelated NVIDIA hardware with "grace" in the marketing name. This is unlikely to cause problems because:
1. If it matches erroneously, it would set HIGH_VRAM, which is generally fine
2. The detection only matters for unified memory systems
3. If NVIDIA releases a non-unified "Grace-something" GPU, it would have discrete VRAM < system RAM, so it would also need explicit detection

**Risk:** Very low. The ratio check (> 0.95) acts as a second factor for any name-based match.

### W4. Some offload functions still return CPU on unified memory

**Functions affected:** `text_encoder_offload_device()` (line 1053), `vae_offload_device()` (line 1122), `intermediate_device()` (line 1105)

These functions check `args.gpu_only` rather than `vram_state`, so they still return CPU on unified memory unless `--gpu-only` is passed. The docker-compose explicitly does NOT pass `--gpu-only` (line 32 comment: "DON'T use --gpu-only - let the unified memory fabric work naturally").

In practice, this is harmless for correctness because:
- `text_encoder_device()` (line 1059) correctly returns GPU under HIGH_VRAM
- The offload device only matters when models are unloaded, and on unified memory "CPU offload" is just a pointer/address space change, not a physical memory copy

But it means models could ping-pong between CPU and GPU address spaces unnecessarily for offload/reload cycles.

**Recommendation:** Consider making these functions aware of `UNIFIED_MEMORY` so they return GPU when appropriate.

---

## INFO

### I1. Detection heuristics are sound for the target platform

On the DGX Spark (GB10, 128GB unified memory):
- `torch.cuda.get_device_properties(0).total_memory` reports ~128GB
- `psutil.virtual_memory().total` reports ~128GB
- `ratio = 128/128 ≈ 1.0 > 0.95` → unified memory detected
- Device name contains "GB10" → name-based detection also triggers

Both detection paths independently confirm unified memory. The ratio check acts as a generic fallback; the name check provides explicit targeting.

### I2. HIGH_VRAM mode behavior chains are correct for unified memory

With `vram_state = VRAMState.HIGH_VRAM`:

| Function | Behavior | Correct for Unified Memory? |
|----------|----------|---------------------------|
| `unet_offload_device()` | Returns GPU | Yes |
| `unet_inital_load_device()` | Returns GPU (HIGH_VRAM in check) | Yes |
| `text_encoder_device()` | Returns GPU (HIGH_VRAM in check) | Yes |
| `free_memory()` soft_empty_cache guard | Skips cache flush (HIGH_VRAM guard) | Yes |
| `load_models_gpu()` lowvram skip | Skips lowvram logic (HIGH_VRAM guard) | Yes |

All the primary memory management paths correctly keep models on GPU when HIGH_VRAM is set. The `unet_inital_load_device()` function (line 953) already treats `HIGH_VRAM` and `SHARED` identically, confirming the design intent.

### I3. `soft_empty_cache()` change is well-motivated but has subtle callers

**File:** `patches/model_management.py`, lines 1845–1854

The patch skips `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` on unified memory because releasing PyTorch's cached allocator blocks back to the OS causes page faults when PyTorch re-allocates from the same physical pool.

The 5 call sites of `soft_empty_cache()`:

| Call site | Line | Guarded by HIGH_VRAM? | OK to skip empty_cache? |
|-----------|------|-----------------------|------------------------|
| `free_memory()` after unloading | 758 | Yes (line 760) | Yes — won't reach this path |
| `free_memory()` torch free > 25% | 763 | Yes (line 760) | Yes — won't reach this path |
| `cleanup_models_gc()` leak detect | 902 | No | Yes — rare, diagnostic path |
| `get_cast_buffer()` >50MB buffer | 1266 | No | Ok — allocator reuses memory |
| `reset_cast_buffers()` streams | 1296 | No | Ok — allocator reuses memory |

The two unguarded callers in `get_cast_buffer()` and `reset_cast_buffers()` are fine — they delete old buffers and want PyTorch to reuse the memory, which the caching allocator will do internally without releasing to the OS.

### I4. Docker compose environment variables are consistent with the patch

**File:** `docker-compose.yml`, lines 40–50

| Variable | Value | Rationale |
|----------|-------|-----------|
| `CUDA_CACHE_DISABLE` | `1` | Disable CUDA kernel cache (unified memory doesn't benefit) |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | **Removed** | Patch handles caching properly; keeping PyTorch allocator ON is more efficient |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Single GPU, no multi-device contention |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | `1` | Force CUDA managed memory to device allocation |
| `OMP_NUM_THREADS` | `20` | Matches 20-core ARM CPU |
| `CUBLAS_WORKSPACE_CONFIG` | `:0:0` | Minimal workspace for cuBLAS |

The removal of `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is correct — that env var disables PyTorch's caching allocator entirely, which would cause massive allocation overhead. The patched `soft_empty_cache()` achieves the same goal (not releasing to OS) without the cost of disabling the allocator.

The `--disable-pinned-memory` flag (line 32) is correct for unified memory — pinning is unnecessary when GPU and CPU share physical memory.

### I5. No `is_wsl()` unified memory check — probably fine

WSL2 with GPU-PV (GPU Paravirtualization) can report VRAM and RAM in ways that might look like unified memory, but the ratio would still be VRAM < RAM (WSL2 reports the physical GPU's VRAM, not unified). No action needed.

### I6. Volume mount strategy

The read-only volume mount at line 70 (`./patches/model_management.py:/opt/ComfyUI/comfy/model_management.py:ro`) is clean — it overlays the patch without modifying the container image. This means:
- `docker compose down && docker compose up` picks up patch changes
- `docker compose build` is not needed for patch iteration
- The patch is source-controlled in the Sparky repo

---

## Other Functions That COULD Be Patched (Not Required)

These are opportunities for further optimization, not bugs:

### `text_encoder_offload_device()` (line 1053)
Returns CPU unless `--gpu-only`. On unified memory, offloading to CPU is pointless. Could check `UNIFIED_MEMORY` and return GPU.

### `vae_offload_device()` (line 1122)
Same issue — returns CPU unless `--gpu-only`. VAE offload to CPU is a no-op on unified memory.

### `intermediate_device()` (line 1105)
Returns CPU unless `--gpu-only`. Intermediate tensors should stay on GPU in unified memory.

### `PINNED_MEMORY` / `MAX_PINNED_MEMORY` (lines 1395–1404)
Pinned memory is pointless on unified memory — CPU and GPU share physical pages. The `--disable-pinned-memory` flag handles this at the CLI level, but the code still allocates `MAX_PINNED_MEMORY`. Could add a guard.

---

## Summary

| Severity | Count | Actionable in this project? |
|----------|-------|-----------------------------|
| CRITICAL | 2 | No — MPS unreachable in CUDA container |
| WARNING | 4 | W2 (2GB reserve) worth monitoring |
| INFO | 6 | I4 env vars well-configured |

**Verdict:** The patch is safe and correct for the DGX Spark target. The CRITICAL items are MPS regressions that are unreachable in this Docker container but would need fixing before upstreaming. The ratio-based detection heuristic is robust — the > 0.95 threshold effectively filters all real-world discrete GPU configurations. The `soft_empty_cache()` change and HIGH_VRAM override are the right approach for unified memory.

**Recommended fixes before upstreaming:**
1. Remove MPS from `_is_unified_memory()` (fixes both C1 and C2)
2. Consider making the 2GB reserve in `maximum_vram_for_weights()` a fraction of total memory
3. Optionally add `UNIFIED_MEMORY` awareness to `text_encoder_offload_device()`, `vae_offload_device()`, and `intermediate_device()`
