# Transformer Implementation for Neural Machine Translation

A complete PyTorch implementation of the Transformer model introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). The implementation provides a clean, modular, and configurable encoder-decoder architecture for sequence-to-sequence tasks, particularly neural machine translation.

## Features

- **Complete Transformer architecture:** Encoder-decoder model with multi-head self-attention and encoder-decoder cross-attention.
- **Learned positional embeddings:** Adds sequence-position information to token embeddings.
- **Masked attention:** Applies padding masks and a causal target mask to prevent the decoder from attending to future tokens.
- **Modular design:** Separate `SelfAttention`, `TransformerBlock`, `Encoder`, `Decoder`, and `Transformer` components.
- **Configurable:** Adjust embedding size, layer count, attention heads, feed-forward expansion, dropout, and maximum sequence length.
- **CPU and GPU ready:** Uses PyTorch device management and runs on CUDA when available.

## Installation

Install PyTorch using the command appropriate for your platform. For a basic CPU installation:

```bash
pip install torch
```

Clone the repository and run the included smoke test:

```bash
git clone https://github.com/Prabinkarki67/transformer.git
cd transformer
python main.py
```

Expected output:

```text
torch.Size([2, 7, 10])
```

## Architecture Overview

The Transformer relies entirely on attention mechanisms, without recurrent or convolutional layers.

### Encoder

The encoder consists of `N` identical `TransformerBlock` layers. Each layer contains:

1. Multi-head self-attention over the source sequence.
2. A position-wise feed-forward network.
3. Residual connections followed by layer normalization.

Source tokens are mapped to embeddings and combined with learned positional embeddings before passing through the encoder stack.

### Decoder

The decoder consists of `N` identical `DecoderBlock` layers. Each layer contains:

1. Masked multi-head self-attention, which prevents attending to future positions.
2. Multi-head cross-attention over the encoder output.
3. A position-wise feed-forward network.
4. Residual connections followed by layer normalization.

The final linear layer maps decoder hidden states to target vocabulary logits.

### Attention Mechanism

Scaled dot-product attention is computed as:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The implementation computes attention for several heads in parallel, allowing the model to capture different relationships between tokens.

## Configuration Parameters

The main `Transformer` constructor accepts the following parameters:

| Parameter | Description | Default |
| --- | --- | ---: |
| `embed_size` | Dimension of token embeddings and hidden states | `256` |
| `num_layers` | Number of encoder and decoder layers | `6` |
| `heads` | Number of attention heads | `8` |
| `forward_expansion` | Feed-forward hidden dimension multiplier | `4` |
| `dropout` | Dropout rate used in transformer blocks | `0` |
| `max_length` | Maximum sequence length supported by positional embeddings | `100` |
| `src_vocab_size` | Source vocabulary size | Required |
| `trg_vocab_size` | Target vocabulary size | Required |
| `src_pad_idx` | Source padding-token index | Required |
| `trg_pad_idx` | Target padding-token index | Required |
| `device` | Device used for tensors and model execution | `cpu` |

## Usage

```python
import torch

from main import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
		src_vocab_size=10_000,
		trg_vocab_size=10_000,
		src_pad_idx=0,
		trg_pad_idx=0,
		embed_size=256,
		num_layers=6,
		heads=8,
		dropout=0.1,
		device=device,
).to(device)

source = torch.randint(0, 10_000, (32, 64), device=device)
target = torch.randint(0, 10_000, (32, 48), device=device)

logits = model(source, target)
print(logits.shape)  # (32, 48, 10_000)
```

For teacher forcing during training, pass the target sequence shifted by one position as the decoder input and use the next-token sequence as labels.

## Best Practices

### Data Preparation

- Use subword tokenization, such as BPE or SentencePiece, for better handling of rare words.
- Add special tokens such as `<sos>`, `<eos>`, `<pad>`, and `<unk>`.
- Filter out sequences that are too long to avoid unnecessary memory use.
- Shuffle training data to improve convergence.

### Training

- Use learning-rate warmup with a schedule such as:

	```text
	lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
	```

- Use gradient clipping, for example `max_norm=1.0`, to reduce the risk of exploding gradients.
- Apply label smoothing to improve model calibration.
- Use mixed-precision training when supported by the available GPU.
- Use gradient accumulation when memory limits the batch size.

### Regularization

- Use dropout rates between `0.1` and `0.3` depending on dataset size and overfitting behavior.
- Consider weight decay, such as L2 regularization, when validation loss diverges from training loss.
- Stop training early when validation performance stops improving.

### Inference and Decoding

- Use beam search with a width of approximately `4` to `5` for higher-quality translation.
- Apply a length penalty to avoid systematically short translations.
- Consider ensembling multiple models when additional inference cost is acceptable.

## Hyperparameter Tuning

- **Embedding size:** `256` to `512` for medium datasets; `512` to `1024` for larger datasets.
- **Number of layers:** `4` to `6` for medium datasets; `6` to `12` for larger datasets.
- **Attention heads:** `8` to `16` is a common range. Ensure that `embed_size` is divisible by `heads`.
- **Batch size:** Use the largest size that fits in memory, with gradient accumulation when needed.



## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*.
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

If you find this repository useful for your projects, consider starring it on GitHub.
