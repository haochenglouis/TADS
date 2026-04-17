"""Unified vLLM engine factory used by Steps 1 and 3."""

from vllm import LLM, SamplingParams


def create_vllm_engine(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = None,
    **kwargs,
) -> LLM:
    """Create a vLLM inference engine with the given configuration.

    Extra kwargs are forwarded to ``vllm.LLM()``.
    """
    init_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    if max_model_len is not None:
        init_kwargs["max_model_len"] = max_model_len
    init_kwargs.update(kwargs)

    print(f"Loading vLLM engine: {model_name}  (tp={tensor_parallel_size}, "
          f"mem={gpu_memory_utilization}, max_len={max_model_len})")
    return LLM(**init_kwargs)


def create_sampling_params(
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    repetition_penalty: float = 1.2,
    stop: list = None,
    **kwargs,
) -> SamplingParams:
    """Create SamplingParams with sensible defaults."""
    params = dict(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=True,
    )
    if stop:
        params["stop"] = stop
    params.update(kwargs)
    return SamplingParams(**params)
