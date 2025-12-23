// Ladder data model
const ladder = [
  {
    id: 'phase1',
    name: 'Phase 1 – Foundations: GPU Architecture & Parallelism',
    tag: 'Weeks 1–2 · Core mental model',
    description:
      'Build an intuition for GPU hardware, threads/blocks, and memory hierarchy before writing serious kernels.',
    groups: [
      {
        id: 'p1-g1',
        title: 'GPU Architecture Basics',
        meta: 'Understand what a GPU actually is',
        topics: [
          {
            id: 'p1-g1-t1',
            title: 'CPU vs GPU: Why GPUs for ML and HPC',
            description:
              'Understand differences in core counts, SIMD/SIMT execution, and why GPUs excel at throughput workloads.',
            article: 'https://developer.nvidia.com/blog/even-easier-introduction-cuda/',
            paper: 'https://dl.acm.org/doi/10.1145/1365490.1365500',
            video: 'https://www.youtube.com/watch?v=-P28LKWTzrI',
            exercise: 'https://leetgpu.com/challenges',
          },
          {
            id: 'p1-g1-t2',
            title: 'GPU Architecture: SMs, warps, cores',
            description:
              'Learn about streaming multiprocessors (SMs), warps (32 threads), occupancy, and wavefronts on AMD.',
            article: 'https://cvw.cac.cornell.edu/gpu-architecture/gpu-characteristics/kernel_sm',
            paper: 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf',
            video: 'https://www.youtube.com/watch?v=61aNaXjAqm0',
            exercise: 'https://leetgpu.com/challenges',
          },
          {
            id: 'p1-g1-t3',
            title: 'Memory hierarchy: registers, shared, global',
            description:
              'Understand latency and bandwidth differences and why coalesced global memory accesses matter.',
            article:
              'https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/',
            paper: 'https://dl.acm.org/doi/10.1145/1366230.1366238',
            video: 'https://www.youtube.com/watch?v=2NgP68iA2pU',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
        ],
      },
      {
        id: 'p1-g2',
        title: 'CUDA Programming Model Essentials',
        meta: 'Threads · Blocks · Grids',
        topics: [
          {
            id: 'p1-g2-t1',
            title: 'Threads, blocks, and grids',
            description:
              'Learn the mapping from problem size to grid/block config and why block size is usually a multiple of 32.',
            article:
              'https://developer.nvidia.com/blog/even-easier-introduction-cuda/',
            paper: 'https://dl.acm.org/doi/10.1145/1365490.1365508',
            video: 'https://www.youtube.com/watch?v=HLL_d41XUJM',
            exercise: 'https://leetgpu.com/challenges',
          },
          {
            id: 'p1-g2-t2',
            title: 'Thread indexing (1D, 2D, 3D)',
            description:
              'Practice computing global thread indices for 1D/2D/3D kernels with blockIdx, threadIdx, blockDim, gridDim.',
            article:
              'https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy',
            video: 'https://www.youtube.com/watch?v=uQBy6jqbmlU',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
          {
            id: 'p1-g2-t3',
            title: 'Hello World CUDA kernel',
            description:
              'Write your first CUDA kernel and launch it with <<<grid, block>>> syntax.',
            article:
              'https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model',
            video: 'https://www.youtube.com/watch?v=IWMJIMzsEMg',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
      {
        id: 'p1-g3',
        title: 'NUMBA & LeetGPU for Quick Experiments',
        meta: 'Fast feedback, zero install',
        topics: [
          {
            id: 'p1-g3-t1',
            title: 'GPU Puzzles with Numba',
            description:
              'Use the GPU Puzzles notebook (Numba CUDA backend) to get interactive practice on core kernel patterns.',
            article: 'https://github.com/srush/GPU-Puzzles',
            video: 'https://www.youtube.com/watch?v=H-3rK2F26nE',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
          {
            id: 'p1-g3-t2',
            title: 'Online CUDA execution with LeetGPU',
            description:
              'Write and run CUDA kernels directly in your browser without needing a local NVIDIA GPU.',
            article: 'https://news.ycombinator.com/item?id=42742290',
            video: 'https://www.youtube.com/results?search_query=leetgpu',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
    ],
  },
  {
    id: 'phase2',
    name: 'Phase 2 – Core Kernels: Maps, Reductions, Scans & MatMul',
    tag: 'Weeks 3–4 · Core kernel patterns',
    description:
      'Learn the key parallel patterns that show up everywhere: maps, reductions, prefix sums, and basic matrix multiply.',
    groups: [
      {
        id: 'p2-g1',
        title: 'Map & Elementwise Kernels',
        meta: 'Add, scale, activation functions',
        topics: [
          {
            id: 'p2-g1-t1',
            title: 'Elementwise map kernels',
            description:
              'Implement a kernel where each thread processes one or more elements (e.g., y[i] = f(x[i])).',
            article:
              'https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/',
            video: 'https://www.youtube.com/watch?v=wlM6RkF4Fqs',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
          {
            id: 'p2-g1-t2',
            title: 'Grid-stride loops',
            description:
              'Use grid-stride loops to write kernels that scale across different GPU sizes and input sizes.',
            article:
              'https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/',
            video: 'https://www.youtube.com/watch?v=faSv2uaTE5c',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
      {
        id: 'p2-g2',
        title: 'Parallel Reductions',
        meta: 'Sum, max, min over arrays',
        topics: [
          {
            id: 'p2-g2-t1',
            title: 'Tree-based block reduction',
            description:
              'Implement a shared-memory reduction within a block for sums, max, etc.',
            article:
              'https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf',
            paper: 'https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf',
            video: 'https://www.youtube.com/watch?v=8CvyCkgCG2Q',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
          {
            id: 'p2-g2-t2',
            title: 'Warp-level primitives (__shfl_sync)',
            description:
              'Use warp shuffle intrinsics to implement efficient warp-level reductions.',
            article:
              'https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/',
            paper: 'https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/',
            video:
              'https://www.youtube.com/watch?v=I0-izyq6q5s',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
      {
        id: 'p2-g3',
        title: 'Prefix Sums & Scans',
        meta: 'Building block for many algorithms',
        topics: [
          {
            id: 'p2-g3-t1',
            title: 'Inclusive/exclusive scan',
            description:
              'Learn how parallel prefix sum (scan) works and why it is so widely useful.',
            article:
              'https://developer.nvidia.com/gpugems/gpugems3/part-vi-simulation-and-numerical-algorithms/chapter-39-parallel-prefix-sum-scan-cuda',
            paper: 'https://dl.acm.org/doi/10.1145/7902.7903',
            video: 'https://www.youtube.com/watch?v=My8XmuP4Q2c',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
          {
            id: 'p2-g3-t2',
            title: 'CUDA scan exercise',
            description:
              'Implement a simple block-level scan using shared memory and test with different sizes.',
            article: 'https://tinkerd.net/blog/machine-learning/cuda-basics/',
            video: 'https://www.youtube.com/watch?v=gY7Zg3y5A1k',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
      {
        id: 'p2-g4',
        title: 'Shared Memory & Tiling (MatMul)',
        meta: 'Foundations of performance',
        topics: [
          {
            id: 'p2-g4-t1',
            title: 'Shared memory tiling for matrices',
            description:
              'Use shared memory tiles to reuse data and reduce global memory bandwidth for matrix multiplication.',
            article:
              'https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/',
            paper: 'https://arxiv.org/abs/1509.02308',
            video:
              'https://www.youtube.com/watch?v=5tDqSTmJBM',
            exercise: 'https://github.com/srush/GPU-Puzzles',
          },
          {
            id: 'p2-g4-t2',
            title: 'Naive vs tiled matmul comparison',
            description:
              'Compare naive matmul with a tiled version and measure the performance difference.',
            article:
              'https://tinkerd.net/blog/machine-learning/cuda-basics/',
            video:
              'https://www.youtube.com/watch?v=kENugLI4M0A',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
    ],
  },
  {
    id: 'phase3',
    name: 'Phase 3 – PyTorch & Custom Kernels',
    tag: 'Weeks 5–6 · Integrate with DL frameworks',
    description:
      'Learn how to extend PyTorch with your own CUDA kernels and profile end-to-end deep learning workloads.',
    groups: [
      {
        id: 'p3-g1',
        title: 'Profiling Deep Learning Workloads',
        meta: 'Find real bottlenecks',
        topics: [
          {
            id: 'p3-g1-t1',
            title: 'PyTorch profiler basics',
            description:
              'Use torch.profiler to capture operator-level performance traces and identify hot spots.',
            article:
              'https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html',
            video: 'https://www.youtube.com/watch?v=LuhJEEJQgUM',
            exercise: 'https://colab.research.google.com',
          },
          {
            id: 'p3-g1-t2',
            title: 'Nsight Systems timeline',
            description:
              'Capture a timeline of CUDA kernels and memory operations for a training step.',
            article:
              'https://developer.nvidia.com/blog/profiling-and-optimizing-deep-learning-models-using-nvidia-nsight-systems/',
            video: 'https://www.youtube.com/watch?v=FbdbglaQxnI',
            exercise: 'https://colab.research.google.com',
          },
        ],
      },
      {
        id: 'p3-g2',
        title: 'Custom CUDA Ops in PyTorch',
        meta: 'C++/CUDA extension, autograd',
        topics: [
          {
            id: 'p3-g2-t1',
            title: 'PyTorch C++/CUDA extension basics',
            description:
              'Build a simple C++/CUDA extension and call it from Python using torch.utils.cpp_extension.',
            article:
              'https://pytorch.org/tutorials/advanced/cpp_extension.html',
            video:
              'https://www.youtube.com/watch?v=LI3h8aVchwo',
            exercise: 'https://colab.research.google.com',
            python: 'https://pytorch.org/cppdist/',
            cpp: 'https://github.com/pytorch/pytorch/tree/main/torch/csrc',
          },
          {
            id: 'p3-g2-t2',
            title: 'Custom autograd.Function',
            description:
              'Wrap your CUDA kernel in a custom autograd.Function with forward and backward methods.',
            article:
              'https://pytorch.org/docs/stable/notes/extending.html',
            video:
              'https://www.youtube.com/watch?v=7smC63E0K-w',
            exercise: 'https://colab.research.google.com',
            python: 'https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function',
          },
        ],
      },
      {
        id: 'p3-g3',
        title: 'Mixed Precision & Memory Optimizations',
        meta: 'AMP, gradient checkpointing',
        topics: [
          {
            id: 'p3-g3-t1',
            title: 'Automatic Mixed Precision (AMP)',
            description:
              'Use torch.cuda.amp.autocast and GradScaler to train models in FP16/BF16 safely.',
            article:
              'https://pytorch.org/docs/stable/amp.html',
            video: 'https://www.youtube.com/watch?v=dXB-KQYkNds',
            exercise: 'https://colab.research.google.com',
          },
          {
            id: 'p3-g3-t2',
            title: 'Gradient checkpointing',
            description:
              'Trade compute for memory by recomputing activations during backward pass.',
            article:
              'https://pytorch.org/docs/stable/checkpoint.html',
            video: 'https://www.youtube.com/watch?v=DcJ5zp0q4GE',
            exercise: 'https://colab.research.google.com',
          },
        ],
      },
    ],
  },
  {
    id: 'phase4',
    name: 'Phase 4 – Triton, Kernel Fusion & torch.compile',
    tag: 'Weeks 7–8 · Higher-level kernel authoring',
    description:
      'Learn Triton for concise high-performance kernels, and use torch.compile to fuse operations in deep learning models.',
    groups: [
      {
        id: 'p4-g1',
        title: 'Triton Basics',
        meta: 'Pythonic GPU kernel language',
        topics: [
          {
            id: 'p4-g1-t1',
            title: 'Triton programming model',
            description:
              'Understand Triton\'s block-level programming model, program_id, and pointer arithmetic.',
            article: 'https://openai.com/index/triton/',
            paper: 'https://arxiv.org/abs/2211.15841',
            video: 'https://www.youtube.com/watch?v=_GfPYLNQ6Jw',
            exercise: 'https://github.com/openai/triton/tree/main/python/tutorials',
            python: 'https://github.com/openai/triton',
            cpp: 'https://github.com/openai/triton/tree/main/include/triton',
          },
          {
            id: 'p4-g1-t2',
            title: 'Writing a Triton matmul',
            description:
              'Implement a basic matrix multiplication kernel in Triton and compare to PyTorch.',
            article:
              'https://triton-lang.org/main/getting-started/tutorials/02-matrix-multiplication.html',
            video:
              'https://www.youtube.com/watch?v=GbPG0pQZntE',
            exercise: 'https://github.com/openai/triton',
            python: 'https://github.com/openai/triton/tree/main/python/tutorials/02-fused-softmax.py',
          },
        ],
      },
      {
        id: 'p4-g2',
        title: 'torch.compile & Operator Fusion',
        meta: 'Compilers for PyTorch',
        topics: [
          {
            id: 'p4-g2-t1',
            title: 'torch.compile basics',
            description:
              'Use torch.compile to automatically optimize and fuse operations in PyTorch models.',
            article:
              'https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html',
            paper: 'https://arxiv.org/abs/2208.06107',
            video: 'https://www.youtube.com/watch?v=8QnvT6j7Cxs',
            exercise: 'https://colab.research.google.com',
            python: 'https://pytorch.org/docs/stable/generated/torch.compile.html',
          },
          {
            id: 'p4-g2-t2',
            title: 'Manual fusion patterns',
            description:
              'Identify common fusion opportunities like LayerNorm + bias + activation and implement fused kernels.',
            article:
              'https://developer.nvidia.com/blog/delivering-the-missing-building-blocks-for-nvidia-cuda-kernel-fusion-in-python/',
            video: 'https://www.youtube.com/watch?v=seGDEWwyhek',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
    ],
  },
  {
    id: 'phase5',
    name: 'Phase 5 – LLM & Modern GPU Concepts',
    tag: 'Weeks 9–10 · FlashAttention, KV cache, quantization',
    description:
      'Focus on the most important GPU concepts behind modern LLM inference and training.',
    groups: [
      {
        id: 'p5-g1',
        title: 'FlashAttention & Attention Kernels',
        meta: 'Fast attention for long sequences',
        topics: [
          {
            id: 'p5-g1-t1',
            title: 'FlashAttention core ideas',
            description:
              'Understand how IO-aware attention reduces memory reads and writes.',
            article: 'https://arxiv.org/abs/2205.14135',
            video: 'https://www.youtube.com/watch?v=eMlx5fFNoYc',
            exercise: 'https://github.com/Dao-AILab/flash-attention',
            python: 'https://github.com/Dao-AILab/flash-attention',
            cpp: 'https://github.com/Dao-AILab/flash-attention/tree/main/csrc',
          },
          {
            id: 'p5-g1-t2',
            title: 'FlashAttention-2 / 3 in practice',
            description:
              'Understand improvements in FA-2 and FA-3 and how they are integrated in PyTorch/vLLM.',
            article: 'https://pytorch.org/blog/flashattention-3/',
            video:
              'https://www.youtube.com/watch?v=5W3ZyW8ndvk',
            exercise: 'https://github.com/Dao-AILab/flash-attention',
            python: 'https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn',
            cpp: 'https://github.com/Dao-AILab/flash-attention/tree/main/csrc',
          },
        ],
      },
      {
        id: 'p5-g2',
        title: 'KV Cache & Long-Context Inference',
        meta: 'Memory bottlenecks in LLMs',
        topics: [
          {
            id: 'p5-g2-t1',
            title: 'Key-Value cache basics',
            description:
              'Learn what the kv-cache stores in transformer decoders and why it dominates memory at long context lengths.',
            article: 'https://www.baseten.co/blog/llm-transformer-inference-guide/',
            video: 'https://www.youtube.com/watch?v=VY1K99HCGqY',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/vllm-project/vllm',
            cpp: 'https://github.com/vllm-project/vllm/tree/main/csrc',
          },
          {
            id: 'p5-g2-t2',
            title: 'Paged KV cache & vLLM',
            description:
              'See how vLLM uses a paged KV cache to reduce fragmentation and improve throughput.',
            article: 'https://docs.vllm.ai/en/latest/overview/architecture.html',
            paper: 'https://arxiv.org/abs/2309.06180',
            video: 'https://www.youtube.com/watch?v=9UjU8pZqIww',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/vllm-project/vllm/tree/main/vllm',
            cpp: 'https://github.com/vllm-project/vllm/tree/main/csrc',
          },
        ],
      },
      {
        id: 'p5-g3',
        title: 'Quantization for LLMs',
        meta: 'INT8 / 4-bit / FP8',
        topics: [
          {
            id: 'p5-g3-t1',
            title: 'Post-training quantization',
            description:
              'Understand how INT8/4-bit quantization works for LLM weights and activations.',
            article: 'https://huggingface.co/docs/transformers/main/en/main_classes/quantization',
            paper: 'https://arxiv.org/abs/2210.17323',
            video: 'https://www.youtube.com/watch?v=G_i8mTvIuLI',
            exercise: 'https://github.com/TimDettmers/bitsandbytes',
            python: 'https://github.com/TimDettmers/bitsandbytes',
            cpp: 'https://github.com/TimDettmers/bitsandbytes/tree/main/csrc',
          },
          {
            id: 'p5-g3-t2',
            title: 'Quantization-aware GPU kernels',
            description:
              'Study quantization-friendly kernels (e.g., QKV matmuls) and their constraints. (Needs more curated resources.)',
            article: '',
            video: '',
            exercise: 'https://leetgpu.com/challenges',
          },
        ],
      },
    ],
  },
  {
    id: 'phase6',
    name: 'Phase 6 – Distributed, Multi-GPU & Advanced Topics',
    tag: 'Weeks 11–12 · DDP, tensor parallelism, gaps',
    description:
      'Learn how to scale across multiple GPUs, and note advanced topics where the ecosystem is still evolving.',
    groups: [
      {
        id: 'p6-g1',
        title: 'Distributed Data Parallel (DDP)',
        meta: 'Multi-GPU training',
        topics: [
          {
            id: 'p6-g1-t1',
            title: 'PyTorch DDP basics',
            description:
              'Set up data-parallel training across multiple GPUs on a single node using torch.distributed.',
            article: 'https://pytorch.org/tutorials/intermediate/ddp_tutorial.html',
            video: 'https://www.youtube.com/watch?v=6F3ONwrCxMg',
            exercise: 'https://colab.research.google.com',
          },
          {
            id: 'p6-g1-t2',
            title: 'NCCL communication patterns',
            description:
              'Understand all-reduce, broadcast, and reduce-scatter operations underpinning DDP.',
            article:
              'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html',
            paper: 'https://arxiv.org/abs/1807.11205',
            video: 'https://www.youtube.com/watch?v=E8c9PhiP210',
            exercise: 'https://github.com/NVIDIA/nccl-tests',
          },
        ],
      },
      {
        id: 'p6-g2',
        title: 'Tensor & Pipeline Parallelism',
        meta: 'Large model training',
        topics: [
          {
            id: 'p6-g2-t1',
            title: 'Tensor/pipeline parallel basics',
            description:
              'Learn how Megatron-LM and DeepSpeed shard large models across multiple GPUs.',
            article: 'https://arxiv.org/abs/2205.05198',
            video: 'https://www.youtube.com/watch?v=0QwZ9BtVu0E',
            exercise: 'https://github.com/NVIDIA/Megatron-LM',
          },
          {
            id: 'p6-g2-t2',
            title: 'Kernel fusion for parallelism',
            description:
              'Study how fused kernels reduce communication overhead. (Needs more curated implementation walk-throughs.)',
            article: '',
            video: '',
            exercise: 'https://github.com/NVIDIA/Megatron-LM',
          },
        ],
      },
      {
        id: 'p6-g3',
        title: 'Emerging Topics & Gaps',
        meta: 'Stay current as ecosystem evolves',
        topics: [
          {
            id: 'p6-g3-t1',
            title: 'GPU optimization for new hardware (H200, AMD, etc.)',
            description:
              'Track best practices for optimizing on new GPU architectures and vendors. (Needs more curated resources.)',
            article: '',
            video: '',
            exercise: 'https://leetgpu.com/challenges',
          },
          {
            id: 'p6-g3-t2',
            title: 'Kernel libraries & kernel hubs',
            description:
              'Explore reusable kernel libraries like Hugging Face Kernel Hub and CUDA kernel collections.',
            article: 'https://huggingface.co/blog/hello-hf-kernels',
            video: '',
            exercise: 'https://github.com/huggingface/hf-kernels',
          },
        ],
      },
    ],
  },
  {
    id: 'phase7',
    name: 'Phase 7 – Generative AI Applications',
    tag: 'Weeks 13–14 · Text, image, audio, video generation',
    description:
      'Apply GPU programming knowledge to modern generative AI applications across different modalities.',
    groups: [
      {
        id: 'p7-g1',
        title: 'Image Generation & Diffusion Models',
        meta: 'Stable Diffusion, DALL-E, GANs',
        topics: [
          {
            id: 'p7-g1-t1',
            title: 'Diffusion model fundamentals',
            description:
              'Understand forward/reverse diffusion processes, noise prediction, and denoising steps.',
            article: 'https://arxiv.org/abs/2006.11239',
            paper: 'https://arxiv.org/abs/2006.11239',
            video: 'https://www.youtube.com/watch?v=HoKDTa5jHvg',
            exercise: 'https://github.com/huggingface/diffusers',
            python: 'https://github.com/huggingface/diffusers/tree/main/src/diffusers',
          },
          {
            id: 'p7-g1-t2',
            title: 'Stable Diffusion pipeline & optimization',
            description:
              'Learn about U-Net architectures, CLIP text encoders, and GPU optimizations for inference.',
            article: 'https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl',
            video: 'https://www.youtube.com/watch?v=0B3i7lM2pDU',
            exercise: 'https://github.com/Stability-AI/stablediffusion',
            python: 'https://github.com/huggingface/diffusers',
          },
          {
            id: 'p7-g1-t3',
            title: 'GANs and adversarial training',
            description:
              'Study Generative Adversarial Networks, discriminator/generator dynamics, and training stability.',
            article: 'https://arxiv.org/abs/1406.2661',
            paper: 'https://arxiv.org/abs/1406.2661',
            video: 'https://www.youtube.com/watch?v=eyxmSmjmNS0',
            exercise: 'https://github.com/NVlabs/stylegan3',
            python: 'https://github.com/NVlabs/stylegan3',
          },
        ],
      },
      {
        id: 'p7-g2',
        title: 'Audio Generation & Synthesis',
        meta: 'Voice synthesis, music generation',
        topics: [
          {
            id: 'p7-g2-t1',
            title: 'Neural vocoders & waveform synthesis',
            description:
              'Learn about HiFi-GAN, WaveNet, and GPU-accelerated audio synthesis techniques.',
            article: 'https://arxiv.org/abs/2010.05646',
            paper: 'https://arxiv.org/abs/1609.03499',
            video: 'https://www.youtube.com/watch?v=8xtNyGljyMQ',
            exercise: 'https://github.com/jik876/hifi-gan',
            python: 'https://github.com/jik876/hifi-gan',
          },
          {
            id: 'p7-g2-t2',
            title: 'Text-to-speech (TTS) systems',
            description:
              'Understand Tacotron, FastSpeech architectures and real-time TTS inference optimization.',
            article: 'https://arxiv.org/abs/1710.08969',
            paper: 'https://arxiv.org/abs/1905.09263',
            video: 'https://www.youtube.com/watch?v=zlEJHt2KB6I',
            exercise: 'https://github.com/coqui-ai/TTS',
            python: 'https://github.com/coqui-ai/TTS',
          },
          {
            id: 'p7-g2-t3',
            title: 'Music generation & audio transformers',
            description:
              'Explore MusicGen, AudioLM, and transformer-based approaches to music synthesis.',
            article: 'https://arxiv.org/abs/2306.05284',
            paper: 'https://arxiv.org/abs/2209.03143',
            video: 'https://www.youtube.com/watch?v=VPv9cK1bGsk',
            exercise: 'https://github.com/facebookresearch/audiocraft',
            python: 'https://github.com/facebookresearch/audiocraft',
          },
        ],
      },
      {
        id: 'p7-g3',
        title: 'Video Generation & Temporal Models',
        meta: 'Video diffusion, frame prediction',
        topics: [
          {
            id: 'p7-g3-t1',
            title: 'Video diffusion models',
            description:
              'Learn about spatio-temporal diffusion, 3D U-Nets, and video generation architectures.',
            article: 'https://arxiv.org/abs/2204.03458',
            paper: 'https://arxiv.org/abs/2211.06512',
            video: 'https://www.youtube.com/watch?v=zcJ2rXcFjSQ',
            exercise: 'https://github.com/guoyww/AnimateDiff',
            python: 'https://github.com/guoyww/AnimateDiff',
          },
          {
            id: 'p7-g3-t2',
            title: 'Frame interpolation & super-resolution',
            description:
              'Study optical flow, frame prediction networks, and real-time video enhancement techniques.',
            article: 'https://arxiv.org/abs/2104.11222',
            paper: 'https://arxiv.org/abs/1711.08162',
            video: 'https://www.youtube.com/watch?v=6kR3Hn9nkTk',
            exercise: 'https://github.com/HolyWu/vs-realesrgan',
            python: 'https://github.com/xinntao/Real-ESRGAN',
          },
          {
            id: 'p7-g3-t3',
            title: 'Long-form video generation',
            description:
              'Explore Sora-style models, temporal consistency, and scaling video generation to minutes.',
            article: 'https://openai.com/research/video-generation-models-as-world-simulators',
            video: 'https://www.youtube.com/watch?v=T-QAlcXR6VQ',
            exercise: 'https://github.com/Picsart-AI-Research/Open-Sora',
            python: 'https://github.com/Picsart-AI-Research/Open-Sora',
          },
        ],
      },
      {
        id: 'p7-g4',
        title: 'Text Generation & LLM Inference',
        meta: 'GPU optimizations for text generation',
        topics: [
          {
            id: 'p7-g4-t1',
            title: 'Transformer inference optimization',
            description:
              'GPU kernels for attention, feed-forward networks, and efficient transformer inference at scale.',
            article: 'https://arxiv.org/abs/2305.19466',
            paper: 'https://arxiv.org/abs/2205.14135',
            video: 'https://www.youtube.com/watch?v=1EL9gJ6J7Bg',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/vllm-project/vllm/tree/main/vllm',
            cpp: 'https://github.com/vllm-project/vllm/tree/main/csrc',
          },
          {
            id: 'p7-g4-t2',
            title: 'Speculative decoding & draft models',
            description:
              'GPU-accelerated speculative execution techniques to improve text generation throughput.',
            article: 'https://arxiv.org/abs/2211.17192',
            video: 'https://www.youtube.com/watch?v=pLRFG3z6TMo',
            exercise: 'https://github.com/deepmind/speculative-decoding',
            python: 'https://github.com/hao-ai-lab/Medusa',
          },
          {
            id: 'p7-g4-t3',
            title: 'Continuous batching & request scheduling',
            description:
              'GPU memory management and scheduling algorithms for serving multiple concurrent text generation requests.',
            article: 'https://www.anyscale.com/blog/continuous-batching-llm-inference',
            video: 'https://www.youtube.com/watch?v=0P5bLU7mfXE',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/vllm-project/vllm/tree/main/vllm/engine',
          },
        ],
      },
      {
        id: 'p7-g5',
        title: 'Multimodal & Cross-Modal Generation',
        meta: 'CLIP, DALL-E, unified models',
        topics: [
          {
            id: 'p7-g5-t1',
            title: 'CLIP and contrastive learning',
            description:
              'Understand vision-language alignment, zero-shot classification, and multimodal embeddings.',
            article: 'https://arxiv.org/abs/2103.00020',
            paper: 'https://arxiv.org/abs/2103.00020',
            video: 'https://www.youtube.com/watch?v=zfQRLL9LtbI',
            exercise: 'https://github.com/openai/CLIP',
            python: 'https://github.com/openai/CLIP',
          },
          {
            id: 'p7-g5-t2',
            title: 'Text-to-image models (DALL-E, Imagen)',
            description:
              'Learn about transformer-based image generation and scaling multimodal architectures.',
            article: 'https://arxiv.org/abs/2205.11487',
            paper: 'https://arxiv.org/abs/2102.12092',
            video: 'https://www.youtube.com/watch?v=qTgPSKKjfVg',
            exercise: 'https://github.com/openai/DALL-E',
            python: 'https://github.com/openai/DALL-E',
          },
          {
            id: 'p7-g5-t3',
            title: 'Unified multimodal architectures',
            description:
              'Explore GPT-4V, LLaVA, and end-to-end multimodal models that handle multiple input/output types.',
            article: 'https://arxiv.org/abs/2304.08485',
            paper: 'https://arxiv.org/abs/2310.03744',
            video: 'https://www.youtube.com/watch?v=3X9jRJkj56Y',
            exercise: 'https://github.com/haotian-liu/LLaVA',
            python: 'https://github.com/haotian-liu/LLaVA',
          },
        ],
      },
      {
        id: 'p7-g6',
        title: 'GPU Optimizations for Generative AI',
        meta: 'Memory, latency, throughput for generation',
        topics: [
          {
            id: 'p7-g6-t1',
            title: 'Memory-efficient generation techniques',
            description:
              'GPU memory optimization for autoregressive generation, including memory-efficient attention and KV-cache management.',
            article: 'https://arxiv.org/abs/2205.14135',
            paper: 'https://arxiv.org/abs/2309.06180',
            video: 'https://www.youtube.com/watch?v=3JZbrgO8c6I',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/Dao-AILab/flash-attention',
          },
          {
            id: 'p7-g6-t2',
            title: 'Low-latency inference kernels',
            description:
              'Custom CUDA kernels for real-time generation, including optimized matmuls and attention for low-latency requirements.',
            article: 'https://developer.nvidia.com/blog/optimizing-gpu-inference-latency-and-throughput/',
            video: 'https://www.youtube.com/watch?v=4a8B9V3X6rI',
            exercise: 'https://github.com/openai/triton',
            python: 'https://github.com/openai/triton/tree/main/python/tutorials',
          },
          {
            id: 'p7-g6-t3',
            title: 'Throughput optimization for generation',
            description:
              'GPU techniques for maximizing generation throughput including dynamic batching, parallel decoding, and request multiplexing.',
            article: 'https://arxiv.org/abs/2309.06180',
            video: 'https://www.youtube.com/watch?v=0P5bLU7mfXE',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/vllm-project/vllm/tree/main/vllm/engine',
          },
        ],
      },
      {
        id: 'p7-g7',
        title: 'Serving & Deployment for Generative AI',
        meta: 'Production deployment on GPUs',
        topics: [
          {
            id: 'p7-g7-t1',
            title: 'Model serving frameworks (vLLM, TensorRT-LLM)',
            description:
              'GPU-optimized serving frameworks for generative AI models including TensorRT optimization and inference engines.',
            article: 'https://docs.vllm.ai/en/latest/',
            video: 'https://www.youtube.com/watch?v=5Xc2bNN3M6g',
            exercise: 'https://github.com/vllm-project/vllm',
            python: 'https://github.com/NVIDIA/TensorRT-LLM',
            cpp: 'https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp',
          },
          {
            id: 'p7-g7-t2',
            title: 'GPU model compilation and optimization',
            description:
              'Torch.compile, TensorRT, and CUDA graph optimizations specifically for generative AI inference pipelines.',
            article: 'https://pytorch.org/docs/stable/torch.compiler.html',
            video: 'https://www.youtube.com/watch?v=7smC63E0K-w',
            exercise: 'https://github.com/pytorch/pytorch/tree/main/torch/csrc/jit',
            python: 'https://developer.nvidia.com/tensorrt',
          },
          {
            id: 'p7-g7-t3',
            title: 'Multi-GPU serving and load balancing',
            description:
              'Distributed serving architectures, GPU memory management across nodes, and load balancing for high-throughput generation.',
            article: 'https://docs.vllm.ai/en/latest/serving/distributed_serving.html',
            video: 'https://www.youtube.com/watch?v=6F3ONwrCxMg',
            exercise: 'https://github.com/vllm-project/vllm/tree/main/examples',
            python: 'https://github.com/vllm-project/vllm/tree/main/vllm/distributed',
          },
        ],
      },
      {
        id: 'p7-g8',
        title: 'Real-Time & Interactive Generation',
        meta: 'Low-latency generation across modalities',
        topics: [
          {
            id: 'p7-g8-t1',
            title: 'Real-time audio generation & processing',
            description:
              'GPU-accelerated real-time audio synthesis, effects processing, and streaming audio generation with low latency.',
            article: 'https://arxiv.org/abs/2104.03502',
            video: 'https://www.youtube.com/watch?v=8xtNyGljyMQ',
            exercise: 'https://github.com/facebookresearch/audiocraft',
            python: 'https://github.com/jik876/hifi-gan',
          },
          {
            id: 'p7-g8-t2',
            title: 'Interactive video generation',
            description:
              'GPU techniques for interactive video editing, real-time style transfer, and user-guided video generation.',
            article: 'https://arxiv.org/abs/2306.07280',
            video: 'https://www.youtube.com/watch?v=zcJ2rXcFjSQ',
            exercise: 'https://github.com/guoyww/AnimateDiff',
            python: 'https://github.com/Picsart-AI-Research/Open-Sora',
          },
          {
            id: 'p7-g8-t3',
            title: 'Human-in-the-loop generation systems',
            description:
              'GPU-optimized interactive systems with real-time feedback loops, user guidance, and collaborative generation workflows.',
            article: 'https://arxiv.org/abs/2308.10574',
            video: 'https://www.youtube.com/watch?v=3X9jRJkj56Y',
            exercise: 'https://github.com/haotian-liu/LLaVA',
            python: 'https://github.com/openai/CLIP',
          },
        ],
      },
      {
        id: 'p7-g9',
        title: 'Model Compression for Generative AI',
        meta: 'Quantization, distillation, pruning for generation',
        topics: [
          {
            id: 'p7-g9-t1',
            title: 'Quantization for generative models',
            description:
              'GPU-accelerated quantization techniques for diffusion models, transformers, and multimodal architectures including FP8, INT8, and 4-bit quantization.',
            article: 'https://arxiv.org/abs/2305.14314',
            paper: 'https://arxiv.org/abs/2210.17323',
            video: 'https://www.youtube.com/watch?v=G_i8mTvIuLI',
            exercise: 'https://github.com/TimDettmers/bitsandbytes',
            python: 'https://github.com/TimDettmers/bitsandbytes',
          },
          {
            id: 'p7-g9-t2',
            title: 'Knowledge distillation for generation',
            description:
              'GPU techniques for distilling large generative models into smaller, faster versions while preserving generation quality.',
            article: 'https://arxiv.org/abs/2210.17323',
            video: 'https://www.youtube.com/watch?v=8QwEnbKqYvI',
            exercise: 'https://github.com/huggingface/distil-whisper',
            python: 'https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation',
          },
          {
            id: 'p7-g9-t3',
            title: 'Efficient generative architectures',
            description:
              'GPU-optimized lightweight architectures for generation including mobile-optimized models, edge deployment, and resource-constrained generation.',
            article: 'https://arxiv.org/abs/2309.17421',
            video: 'https://www.youtube.com/watch?v=8QwEnbKqYvI',
            exercise: 'https://github.com/huggingface/optimum',
            python: 'https://github.com/huggingface/optimum/tree/main/optimum',
          },
        ],
      },
    ],
  },
];
