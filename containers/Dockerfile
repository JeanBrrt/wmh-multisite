# wmh-multisite container (ROADMAP step 9.4, J-062).
#
# One recipe, two images, chosen by UV_EXTRAS (the optional dependency groups of pyproject.toml):
#   CPU image (default): preprocessing, QC indicators, thresholding, evaluation, biomarkers,
#     harmonisation and the Snakemake workflow. No PyTorch: small, built by the CI on every push.
#       docker build -f containers/Dockerfile -t wmh-multisite .
#   GPU image: adds nnU-Net and HD-BET (PyTorch with CUDA on Linux) and the project's trainers.
#       docker build -f containers/Dockerfile --build-arg UV_EXTRAS="--extra harmonize --extra workflow --extra seg" -t wmh-multisite:gpu .
#
# The environment is the locked one (uv.lock): same versions as the development machine and the CI.
# Data are never copied in the image (.dockerignore); mount them at run time:
#   docker run --rm -v "$PWD/data:/opt/wmh-multisite/data" -v "$PWD/results:/opt/wmh-multisite/results" \
#     wmh-multisite snakemake -s workflow/Snakefile -n

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ARG UV_EXTRAS="--extra harmonize --extra workflow"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MPLBACKEND=Agg \
    PATH="/opt/wmh-multisite/.venv/bin:$PATH"

WORKDIR /opt/wmh-multisite

# 1. Dependencies only (cached layer: rebuilt only when the lock file changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev $UV_EXTRAS

# 2. The project itself
COPY src ./src
COPY config ./config
COPY workflow ./workflow
COPY scripts ./scripts
COPY hpc ./hpc
RUN uv sync --frozen --no-dev $UV_EXTRAS \
 && if python -c "import nnunetv2" 2>/dev/null; then \
        cp hpc/wmh_multisite_trainers.py \
           "$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/"; \
    fi

# nnU-Net folders (used by the GPU image), inside the mounted data folder
ENV nnUNet_raw=/opt/wmh-multisite/data/nnunet/nnUNet_raw \
    nnUNet_preprocessed=/opt/wmh-multisite/data/nnunet/nnUNet_preprocessed \
    nnUNet_results=/opt/wmh-multisite/data/nnunet/nnUNet_results \
    TEMPLATEFLOW_HOME=/opt/wmh-multisite/data/templateflow

CMD ["python", "-c", "import wmh_multisite; print('wmh-multisite ready; see workflow/Snakefile')"]
