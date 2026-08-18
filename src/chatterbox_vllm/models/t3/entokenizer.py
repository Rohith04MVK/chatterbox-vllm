import logging
from pathlib import Path
from typing import ClassVar, List, Optional, Union

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer(PreTrainedTokenizer):
    """
    A VLLM-compatible tokenizer that wraps the original EnTokenizer implementation.
    """
    model_input_names = ["input_ids", "attention_mask"]
    _model_dir: ClassVar[Optional[Path]] = None

    @classmethod
    def set_model_dir(cls, model_dir: Optional[Union[str, Path]]) -> None:
        cls._model_dir = Path(model_dir) if model_dir is not None else None

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.check_vocabset_sot_eot()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *args, **kwargs):
        """
        Instantiate a tokenizer from a pretrained model or path.
        
        Args:
            pretrained_model_name_or_path: Path to the tokenizer file or model name
            **kwargs: Additional arguments to pass to the tokenizer
        """
        kwargs.pop("revision", None)
        kwargs.pop("download_dir", None)
        kwargs.pop("trust_remote_code", None)

        candidates = []
        if cls._model_dir is not None:
            candidates.append(cls._model_dir / "tokenizer.json")
        if pretrained_model_name_or_path:
            path = Path(pretrained_model_name_or_path)
            if path.is_dir():
                candidates.append(path / "tokenizer.json")
            elif path.is_file():
                candidates.append(path)
        candidates.append(Path(__file__).resolve().parent / "tokenizer.json")

        vocab_file = next((str(path) for path in candidates if path.is_file()), None)
        if vocab_file is None:
            raise FileNotFoundError(
                "Could not find tokenizer.json in the model directory or package data."
            )
        return cls(vocab_file=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = text.replace(' ', SPACE)
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(UNK, '')
        return text
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def max_token_id(self) -> int:
        return max(self.tokenizer.get_vocab().values())