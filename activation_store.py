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
        to_cpu = self.data_config.buffer_on_cpu
        all_inputs = []
        all_outputs = []
        for _ in range(self.data_config.num_batches_in_buffer):
            input_ids, attention_mask = self.get_batch_tokens()
            input_acts, output_acts = self.get_activations(input_ids, attention_mask)
            if attention_mask is not None:
                # Only keep activations for real (non-padding) tokens
                mask = attention_mask.bool().unsqueeze(-1)  # (batch, seq, 1)
                inp = input_acts[mask.expand_as(input_acts)].reshape(-1, self.input_size)
                out = output_acts[mask.expand_as(output_acts)].reshape(-1, self.output_size)
            else:
                # Pre-tokenized: no padding, keep all tokens
                inp = input_acts.reshape(-1, self.input_size)
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


class RandomActivationStore:
    """Generates random vectors in activation space, calibrated to real statistics.

    Samples input vectors from a per-dimension Gaussian fitted to real activations,
    then passes them through the actual output module (e.g. MLP) to produce targets.
    The transcoder learns the module's real input-output function, but on inputs
    that would never arise from natural text.

    Call calibrate() before training to fit the Gaussian to real activation stats.
    Does not support CE performance evaluation — use eval_stores for that.
    """

    def __init__(
        self,
        output_module: nn.Module,
        input_size: int,
        output_size: int,
        batch_size: int,
        device: str,
    ):
        self.output_module = output_module
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device
        self.input_mean: torch.Tensor | None = None
        self.input_std: torch.Tensor | None = None

    def calibrate(self, activation_store: "ActivationsStore", n_batches: int = 10):
        """Fit per-dimension Gaussian to real input activations."""
        all_inputs = []
        for _ in range(n_batches):
            x_in, _ = activation_store.next_batch()
            all_inputs.append(x_in)
        all_inputs = torch.cat(all_inputs)
        self.input_mean = all_inputs.mean(dim=0).to(self.device)
        self.input_std = all_inputs.std(dim=0).to(self.device)
        print(
            f"Calibrated RandomActivationStore: "
            f"mean norm={self.input_mean.norm():.2f}, "
            f"mean std={self.input_std.mean():.4f}"
        )

    @torch.no_grad()
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate calibrated random inputs and compute module outputs."""
        assert self.input_mean is not None, "Must call calibrate() first"
        x_in = torch.randn(self.batch_size, self.input_size, device=self.device)
        x_in = x_in * self.input_std + self.input_mean
        y_target = self.output_module(x_in)
        return x_in, y_target


class RandomTokenActivationStore(ActivationsStore):
    """Activation store that feeds uniformly sampled random tokens through the model.

    Since the model can't memorize random data, any features a transcoder learns
    must reflect real model mechanisms rather than dataset-specific patterns.

    Each sequence starts with a BOS token, followed by tokens sampled uniformly
    from the full vocabulary. No attention mask is needed (no padding).
    """

    def _setup_dataset(self):
        """No dataset needed — just store the tokenizer."""
        cfg = self.data_config
        self.tokenizer = cfg.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_batch_tokens(self) -> tuple[torch.Tensor, None]:
        """Generate a batch of random token sequences, each starting with BOS."""
        cfg = self.data_config
        vocab_size = self.tokenizer.vocab_size
        bos_token_id = self.tokenizer.bos_token_id

        random_ids = torch.randint(
            0, vocab_size, (cfg.model_batch_size, cfg.seq_len), device=cfg.device,
        )
        if bos_token_id is not None:
            random_ids[:, 0] = bos_token_id

        return random_ids, None


class MultiLayerActivationsStore:
    """Activation store that hooks into multiple (input, output) module pairs.

    Used for Cross-Layer Transcoders. Each forward pass captures activations at
    every layer simultaneously, yielding aligned per-layer batches.

    Args:
        model: The transformer model to hook into.
        input_modules: Modules whose outputs are the CLT encoder inputs (one per layer).
        output_modules: Modules whose outputs are the CLT reconstruction targets (one per layer).
        data_config: Configuration for data loading.
        input_size: Per-layer input activation dimension.
        output_size: Per-layer output activation dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        input_modules: list[nn.Module],
        output_modules: list[nn.Module],
        data_config: DataConfig,
        input_size: int,
        output_size: int,
    ):
        assert len(input_modules) == len(output_modules)
        self.model = model
        self.output_modules = output_modules
        self.data_config = data_config
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = len(input_modules)

        # Activation storage (written by hooks during forward pass)
        self._input_acts: list[torch.Tensor | None] = [None] * self.n_layers
        self._output_acts: list[torch.Tensor | None] = [None] * self.n_layers

        # Register hooks
        for i, mod in enumerate(input_modules):
            mod.register_forward_hook(self._make_hook(self._input_acts, i))
        for i, mod in enumerate(output_modules):
            mod.register_forward_hook(self._make_hook(self._output_acts, i))

        # Setup dataset
        self._setup_dataset()

        # Buffer state (initialized on first next_batch call)
        self.input_buffers: list[torch.Tensor] | None = None
        self.output_buffers: list[torch.Tensor] | None = None
        self.dataloader: DataLoader | None = None
        self.dataloader_iter: iter | None = None

    @staticmethod
    def _make_hook(storage: list, idx: int):
        def hook(module, input, output):
            storage[idx] = output.detach()
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
        """Get a batch of tokens. Returns (input_ids, attention_mask)."""
        if self.data_config.is_tokenized:
            return self._get_batch_tokens_pretokenized()
        return self._get_batch_tokens_text()

    def _get_batch_tokens_text(self) -> tuple[torch.Tensor, torch.Tensor]:
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
            batch_ids.append(ids[: cfg.seq_len])
        return torch.stack(batch_ids).to(cfg.device), None

    @torch.no_grad()
    def get_activations(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run forward pass and return per-layer (inputs, outputs) activations."""
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        self.model(input_ids, **kwargs)
        assert all(a is not None for a in self._input_acts)
        assert all(a is not None for a in self._output_acts)
        return list(self._input_acts), list(self._output_acts)

    def _fill_buffer(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Fill buffers with activations from multiple forward passes."""
        to_cpu = self.data_config.buffer_on_cpu
        all_inputs = [[] for _ in range(self.n_layers)]
        all_outputs = [[] for _ in range(self.n_layers)]

        for _ in range(self.data_config.num_batches_in_buffer):
            input_ids, attention_mask = self.get_batch_tokens()
            input_acts, output_acts = self.get_activations(input_ids, attention_mask)

            for i in range(self.n_layers):
                if attention_mask is not None:
                    mask = attention_mask.bool().unsqueeze(-1)
                    inp = input_acts[i][mask.expand_as(input_acts[i])].reshape(-1, self.input_size)
                    out = output_acts[i][mask.expand_as(output_acts[i])].reshape(-1, self.output_size)
                else:
                    inp = input_acts[i].reshape(-1, self.input_size)
                    out = output_acts[i].reshape(-1, self.output_size)
                all_inputs[i].append(inp.cpu() if to_cpu else inp)
                all_outputs[i].append(out.cpu() if to_cpu else out)

        return (
            [torch.cat(buf) for buf in all_inputs],
            [torch.cat(buf) for buf in all_outputs],
        )

    def _get_dataloader(self) -> DataLoader:
        """Create dataloader from buffers. All layers are aligned by token position."""
        all_tensors = self.input_buffers + self.output_buffers
        return DataLoader(
            TensorDataset(*all_tensors),
            batch_size=self.data_config.train_batch_size,
            shuffle=True,
        )

    def next_batch(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get next aligned batch of per-layer (inputs, targets) activations."""
        try:
            batch = next(self.dataloader_iter)
        except (StopIteration, TypeError):
            self.input_buffers, self.output_buffers = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)

        n = self.n_layers
        inputs = list(batch[:n])
        targets = list(batch[n:])
        if self.data_config.buffer_on_cpu:
            device = self.data_config.device
            inputs = [x.to(device) for x in inputs]
            targets = [x.to(device) for x in targets]
        return inputs, targets
