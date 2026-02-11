from dataclasses import dataclass

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DataConfig:
    """Config for activation store data loading."""

    dataset_name: str
    tokenizer: PreTrainedTokenizerBase
    text_column: str = "text"
    seq_len: int = 512
    model_batch_size: int = 64
    train_batch_size: int = 4096
    num_batches_in_buffer: int = 10
    device: str = "cuda"
    seed: int = 42
    streaming: bool = True
    lowercase: bool = False
    is_tokenized: bool = False
    token_column: str = "input_ids"
    buffer_on_cpu: bool = False


class ActivationsStore:
    """Activation store using PyTorch hooks. Works with any nn.Module.

    Uses standard PyTorch forward hooks to capture activations from any model.
    Buffers activations from multiple forward passes for efficient training.

    For SAE training (input == target), pass the same module for both
    input_module and output_module.

    Args:
        model: Any PyTorch model
        input_module: Module whose output is the encoder input
        output_module: Module whose output is the encoder target
        data_config: Configuration for data loading
        input_size: Dimension of input activations
        output_size: Dimension of output activations
    """

    def __init__(
        self,
        model: nn.Module,
        input_module: nn.Module,
        output_module: nn.Module,
        data_config: DataConfig,
        input_size: int,
        output_size: int,
    ):
        self.model = model
        self.output_module = output_module
        self.data_config = data_config
        self.input_size = input_size
        self.output_size = output_size

        # Activation storage (temporary, written by hooks)
        self._input_acts: torch.Tensor | None = None
        self._output_acts: torch.Tensor | None = None

        # Register hooks
        input_module.register_forward_hook(self._input_hook)
        output_module.register_forward_hook(self._output_hook)

        # Setup dataset
        self._setup_dataset()

        # Buffer state (initialized on first next_batch call)
        self.input_buffer: torch.Tensor | None = None
        self.output_buffer: torch.Tensor | None = None
        self.dataloader: DataLoader | None = None
        self.dataloader_iter: iter | None = None

    def _input_hook(self, module, input, output):
        self._input_acts = output.detach()

    def _output_hook(self, module, input, output):
        self._output_acts = output.detach()

    def _setup_dataset(self):
        cfg = self.data_config
        self.tokenizer = cfg.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = load_dataset(cfg.dataset_name, split="train", streaming=cfg.streaming)
        self.dataset = self.dataset.shuffle(seed=cfg.seed, buffer_size=10000)
        self.data_iter = iter(self.dataset)

    def get_batch_tokens(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get a batch of tokens. Returns (input_ids, attention_mask).

        For pre-tokenized data, attention_mask is None (no padding).
        """
        if self.data_config.is_tokenized:
            return self._get_batch_tokens_pretokenized()
        return self._get_batch_tokens_text()

    def _get_batch_tokens_text(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of (input_ids, attention_mask) by tokenizing text on-the-fly."""
        cfg = self.data_config
        texts = []
        for _ in range(cfg.model_batch_size):
            try:
                sample = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            text = sample[cfg.text_column]
            if cfg.lowercase:
                text = text.lower()
            texts.append(text)

        tokens = self.tokenizer(
            texts,
            truncation=True,
            max_length=cfg.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            tokens["input_ids"].to(cfg.device),
            tokens["attention_mask"].to(cfg.device),
        )

    def _get_batch_tokens_pretokenized(self) -> tuple[torch.Tensor, None]:
        """Get a batch of input_ids from a pre-tokenized dataset. No padding needed."""
        cfg = self.data_config
        batch_ids = []
        for _ in range(cfg.model_batch_size):
            try:
                sample = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            ids = sample[cfg.token_column]
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long)
            batch_ids.append(ids[:cfg.seq_len])
        return torch.stack(batch_ids).to(cfg.device), None

    @torch.no_grad()
    def get_activations(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and return (input, output) activations."""
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        self.model(input_ids, **kwargs)
        assert self._input_acts is not None and self._output_acts is not None
        return self._input_acts, self._output_acts

    def _fill_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill buffer with activations from multiple forward passes."""
        all_inputs = []
        all_outputs = []
        for _ in range(self.data_config.num_batches_in_buffer):
            input_ids, attention_mask = self.get_batch_tokens()
            input_acts, output_acts = self.get_activations(input_ids, attention_mask)
            if attention_mask is not None:
                # Only keep activations for real (non-padding) tokens
                mask = attention_mask.bool().unsqueeze(-1)  # (batch, seq, 1)
                all_inputs.append(input_acts[mask.expand_as(input_acts)].reshape(-1, self.input_size))
                all_outputs.append(output_acts[mask.expand_as(output_acts)].reshape(-1, self.output_size))
            else:
                # Pre-tokenized: no padding, keep all tokens
                all_inputs.append(input_acts.reshape(-1, self.input_size))
                all_outputs.append(output_acts.reshape(-1, self.output_size))
        inputs = torch.cat(all_inputs, dim=0)
        outputs = torch.cat(all_outputs, dim=0)
        if self.data_config.buffer_on_cpu:
            return inputs.cpu(), outputs.cpu()
        return inputs, outputs

    def _get_dataloader(self) -> DataLoader:
        """Create dataloader from buffers."""
        return DataLoader(
            TensorDataset(self.input_buffer, self.output_buffer),
            batch_size=self.data_config.train_batch_size,
            shuffle=True,
        )

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input, target) activations from buffer."""
        try:
            batch = next(self.dataloader_iter)
        except (StopIteration, TypeError):
            self.input_buffer, self.output_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        x_in, y_target = batch[0], batch[1]
        if self.data_config.buffer_on_cpu:
            return x_in.to(self.data_config.device), y_target.to(self.data_config.device)
        return x_in, y_target
