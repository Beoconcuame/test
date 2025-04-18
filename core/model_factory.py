import sys
import inspect
import logging
import importlib
import torch

from model.ebilstm import EnhancedBiLSTMClassifier
from model.bilstm import BiLSTMClassifier
from model.bigru import BiGRUClassifier
from model.textcnn import TextCNN
from model.transformer import TransformerClassifier

logger = logging.getLogger(__name__)

standard_models = {
    "bilstm": BiLSTMClassifier,
    "bigru": BiGRUClassifier,
    "textcnn": TextCNN,
    "ebilstm": EnhancedBiLSTMClassifier,
    "transformer": TransformerClassifier,
}

ipa_models = {
    "transformer": ("model_ipa.ipa_transformer", "TransformerWithCombinedPositionalEncoding"),
    "bilstm":      ("model_ipa.ipa_bilstm",     "IpaBiLSTMClassifierWithPos"),
    "bigru":      ("model_ipa.ipa_bigru",     "IpaBiGRUClassifierWithPos"),
    "textcnn":    ("model_ipa.ipa_textcnn",   "IpaTextCNNClassifierWithPos"),
}

def get_model(config: dict, vocab_size: int, num_classes: int, device: torch.device) -> torch.nn.Module:

    model_name = config.get('model')
    if not model_name:
        logger.error("'model' must be specified in config.")
        sys.exit(1)

    model_params = config.get('model_params', {})
    init_args = {'vocab_size': vocab_size, 'num_classes': num_classes, **model_params}

    if model_name.endswith('_ipa'):
        base = model_name[:-4]
        module_path, class_name = ipa_models.get(base, (None, None))
        if not class_name:
            logger.error(f"Unknown IPA model '{model_name}'.")
            sys.exit(1)
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            logger.info(f"Loaded IPA model class {class_name} from {module_path}.")
        except (ImportError, AttributeError) as e:
            logger.error(f"Cannot import {class_name} from {module_path}: {e}")
            sys.exit(1)
    else:
        model_class = standard_models.get(model_name)
        if not model_class:
            logger.error(f"Unknown model '{model_name}'. Available: {list(standard_models)}")
            sys.exit(1)
        logger.info(f"Using standard model class {model_class.__name__}.")

    sig = inspect.signature(model_class.__init__)
    valid_args = {k: v for k, v in init_args.items() if k in sig.parameters}
    invalid = set(init_args) - set(valid_args) - {'self'}
    if invalid:
        logger.warning(f"Ignoring unexpected init params: {invalid}")

    if model_name.endswith('_ipa'):
        if 'custom_pos_vocab_size' in sig.parameters and 'custom_pos_vocab_size' not in valid_args:
            logger.error("Missing required 'custom_pos_vocab_size' for IPA model.")
            sys.exit(1)

    try:
        model = model_class(**valid_args)
    except TypeError as e:
        logger.error(f"Error initializing {model_class.__name__}: {e}")
        sys.exit(1)

    logger.info(f"Model {model_name} initialized.")
    return model.to(device)
