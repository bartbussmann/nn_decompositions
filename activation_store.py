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

    For cross-layer transcoders, pass lists of input and output modules.
    Multiple input modules are concatenated along the feature dim so the
    encoder sees all layers' inputs as a single flat vector.

    Args:
        model: Any PyTorch model
        input_module: Module(s) whose output is the encoder input.
            Single module for standard SAE/transcoder, list for CLT.
            Multiple inputs are concatenated along the last dim.
        output_module: Module(s) whose output is the encoder target.
            Single module for standard SAE/transcoder, list for CLT.
        data_config: Configuration for data loading
        input_size: Total dimension of input activations (sum across layers if multi-input)
        output_size: Dimension of output activations (per layer)
    """

    def __init__(
        self,
        model: nn.Module,
        input_module: nn.Module | list[nn.Module],
        output_module: nn.Module | list[nn.Module],
        data_config: DataConfig,
        input_size: int,
        output_size: int,
    ):
        self.model = model
        if isinstance(input_module, list):
            self.input_modules = input_module
        else:
            self.input_modules = [input_module]
        if isinstance(output_module, list):
            self.output_modules = output_module
        else:
            self.output_modules = [output_module]
        self.num_input_layers = len(self.input_modules)
        self.num_output_layers = len(self.output_modules)
        self.data_config = data_config
        self.input_size = input_size
        self.output_size = output_size

        # Activation storage (temporary, written by hooks)
        self._input_acts_list: list[torch.Tensor | None] = [None] * self.num_input_layers
        self._output_acts_list: list[torch.Tensor | None] = [None] * self.num_output_layers

        # Register hooks
        for i, mod in enumerate(self.input_modules):
            mod.register_forward_hook(self._make_input_hook(i))
        for i, mod in enumerate(self.output_modules):
            mod.register_forward_hook(self._make_output_hook(i))

        # Setup dataset
        self._setup_dataset()

        # Buffer state (initialized on first next_batch call)
        self.input_buffer: torch.Tensor | None = None
        self.output_buffer: torch.Tensor | None = None
        self.dataloader: DataLoader | None = None
        self.dataloader_iter: iter | None = None

    @property
    def output_module(self) -> nn.Module:
        """First output module (for single-layer compat and perf eval)."""
        return self.output_modules[0]

    def _make_input_hook(self, idx: int):
        def hook(module, input, output):
            self._input_acts_list[idx] = output.detach()
        return hook

    def _make_output_hook(self, idx: int):
        def hook(module, input, output):
            self._output_acts_list[idx] = output.detach()
        return hook

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
        """Run forward pass and return (input, output) activations.

        Input: single module → (batch, seq, d), multiple → (batch, seq, sum(d_i))
            (concatenated along last dim).
        Output: single module → (batch, seq, d), multiple → (batch, seq, num_layers, d).
        """
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        self.model(input_ids, **kwargs)
        assert all(a is not None for a in self._input_acts_list)
        assert all(a is not None for a in self._output_acts_list)

        if self.num_input_layers == 1:
            input_acts = self._input_acts_list[0]
        else:
            input_acts = torch.cat(self._input_acts_list, dim=-1)

        if self.num_output_layers == 1:
            output_acts = self._output_acts_list[0]
        else:
            output_acts = torch.stack(self._output_acts_list, dim=-2)

        return input_acts, output_acts

    def _fill_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill buffer with activations from multiple forward passes."""
        to_cpu = self.data_config.buffer_on_cpu
        all_inputs = []
        all_outputs = []
        multi = self.num_output_layers > 1
        for _ in range(self.data_config.num_batches_in_buffer):
            input_ids, attention_mask = self.get_batch_tokens()
            input_acts, output_acts = self.get_activations(input_ids, attention_mask)
            if attention_mask is not None:
                # Only keep activations for real (non-padding) tokens
                mask = attention_mask.bool().unsqueeze(-1)  # (batch, seq, 1)
                inp = input_acts[mask.expand_as(input_acts)].reshape(-1, self.input_size)
                if multi:
                    # output_acts: (batch, seq, num_layers, output_size)
                    out_mask = mask.unsqueeze(-1)  # (batch, seq, 1, 1)
                    out = output_acts[out_mask.expand_as(output_acts)].reshape(
                        -1, self.num_output_layers, self.output_size
                    )
                else:
                    out = output_acts[mask.expand_as(output_acts)].reshape(
                        -1, self.output_size
                    )
            else:
                # Pre-tokenized: no padding, keep all tokens
                inp = input_acts.reshape(-1, self.input_size)
                if multi:
                    out = output_acts.reshape(-1, self.num_output_layers, self.output_size)
                else:
                    out = output_acts.reshape(-1, self.output_size)
            all_inputs.append(inp.cpu() if to_cpu else inp)
            all_outputs.append(out.cpu() if to_cpu else out)
        return torch.cat(all_inputs, dim=0), torch.cat(all_outputs, dim=0)

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
