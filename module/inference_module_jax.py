from datasets import load_dataset


from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

pipeline = FlaxWhisperPipline(
    "openai/whisper-large-v2", dtype=jnp.bfloat16, batch_size=16
)

text = pipeline("/content/uploads/d845d3fdfb24f307e5e66aae7f31c811179d6a54.ogg")

print(text)
