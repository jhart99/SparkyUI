# CUDA 13.0 for Blackwell GB10 (sm_121 / compute_121)
# CUDA 12.8 only supports up to sm_120, but GB10 is sm_121.
# "devel" includes nvcc so we can compile CUDA extensions like SageAttention.
FROM nvcr.io/nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_TAG=v0.23.0
ARG SAGEATTN_REF=main
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV CUDA_HOME=/usr/local/cuda
ENV VENV=/opt/venv

ADD https://raw.githubusercontent.com/Comfy-Org/ComfyUI/refs/tags/v0.23.0/requirements.txt /opt/ComfyUI/
RUN ls -l /opt/ComfyUI


# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential ninja-build cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create venv (keeps python deps isolated inside container)
RUN python3 -m venv $VENV
ENV PATH="$VENV/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu132 \
        torch torchvision && \
    pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt

# ---- SageAttention ----
# GB10 is compute capability 12.1 (sm_121).
# CUDA 13.0 NVCC supports sm_121, so we compile directly for it.

# Build/install SageAttention from repo with sm_121 support
RUN CMAKE_BUILD_PARALLEL_LEVEL=8 MAKEFLAGS="-j8" pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/thu-ml/SageAttention@${SAGEATTN_REF}" || true


FROM nvcr.io/nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04 AS runner
ARG COMFYUI_TAG=v0.23.0
ENV TORCH_CUDA_ARCH_LIST="12.1"
ENV CUDA_HOME=/usr/local/cuda

# Base system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3 python3-pip python3-venv python3-dev \
    build-essential ninja-build cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
# Expose the venv binaries to the PATH
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install ComfyUI
ADD https://github.com/comfyanonymous/ComfyUI.git#${COMFYUI_TAG} /opt/ComfyUI/

# Expose ComfyUI
EXPOSE 8188

# Entry script handles runtime updates / flags
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
