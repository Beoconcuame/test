import torch
import sys
import inspect
import logging

logger = logging.getLogger(__name__)

try:
    from model.ebilstm import EnhancedBiLSTMClassifier
    from model.bilstm import BiLSTMClassifier
    from model.bigru import BiGRUClassifier
    from model.textcnn import TextCNN
    from model.transformer import TransformerClassifier
except ImportError as e:
    logger.error(f"CRITICAL: Error importing standard model classes from 'model' directory: {e}. Check paths and file existence.")
    sys.exit(1)

standard_models = {
    "bilstm": BiLSTMClassifier,
    "bigru": BiGRUClassifier,
    "textcnn": TextCNN,
    "ebilstm": EnhancedBiLSTMClassifier,
    "transformer": TransformerClassifier,
}

def get_model(config, vocab_size, num_classes, device):
    """
    Create and return a model instance based on the configuration.
    Handles both standard models and IPA models.
    
    For IPA models the config model name must have a '_ipa' suffix.
    The module is then loaded dynamically from a file named 'ipa_<base_model>.py'
    and the corresponding class (defined via a predefined mapping) is used for initialization.
    
    Args:
        config (dict): Configuration dictionary.
        vocab_size (int): The vocabulary size.
        num_classes (int): Number of classes for classification.
        device (torch.device): The device (CPU or GPU) to move the model to.
    
    Returns:
        torch.nn.Module: The initialized model placed on the specified device.
    """
    model_name = config.get('model')
    if not model_name:
        logger.error("CRITICAL: 'model' key not found or is empty in the configuration.")
        sys.exit(1)

    model_params = config.get('model_params', {})
    logger.info(f"Attempting to create model specified in config: '{model_name}'")
    model = None

    if model_name.endswith("_ipa"):
        ipa_model_base_name = model_name[:-4]
        module_path = f"model_ipa.ipa_{ipa_model_base_name}"
        class_name = None
        if ipa_model_base_name == "transformer":
            class_name = "TransformerWithCombinedPositionalEncoding"
        elif ipa_model_base_name == "bilstm":
            class_name = "IpaBiLSTMClassifierWithPos"
        elif ipa_model_base_name == "bigru":
            class_name = "IpaBiGRUClassifierWithPos"
        elif ipa_model_base_name == "textcnn":
            class_name = "IpaTextCNNClassifierWithPos"
        else:
            logger.error(f"CRITICAL: Unknown IPA model base name '{ipa_model_base_name}' derived from config model name '{model_name}'.")
            sys.exit(1)

        logger.info(f"Identified as IPA model. Attempting to load class '{class_name}' from module '{module_path}'.")
        logger.warning(f"Ensure the Python file corresponding to this module exists: '{module_path.replace('.', '/')}.py'")
        try:
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            logger.info(f"Successfully loaded IPA model class: {model_class.__name__}")
            init_args = {
                'vocab_size': vocab_size,
                'num_classes': num_classes,
                **model_params
            }
            sig = inspect.signature(model_class.__init__)
            valid_args = {k: v for k, v in init_args.items() if k in sig.parameters}
            invalid_args = {k: v for k, v in init_args.items() if k not in sig.parameters and k != 'self'}
            if invalid_args:
                logger.warning(f"Ignoring parameters not accepted by {class_name}.__init__: {invalid_args}")
            if 'custom_pos_vocab_size' not in valid_args:
                logger.error(f"CRITICAL: Missing required parameter 'custom_pos_vocab_size' for IPA model '{model_name}'. Check 'model_params' in config.")
                sys.exit(1)
            logger.info(f"Initializing {class_name} with parameters: {valid_args}")
            model = model_class(**valid_args)
        except ImportError:
            logger.error(f"CRITICAL: ImportError - Could not import class '{class_name}' from module '{module_path}'. Check file existence ('{module_path.replace('.', '/')}.py') and dependencies.")
            sys.exit(1)
        except AttributeError:
            logger.error(f"CRITICAL: AttributeError - Class '{class_name}' not found in module '{module_path}'. Check class definition inside the file.")
            sys.exit(1)
        except TypeError as e:
            logger.error(f"CRITICAL: TypeError initializing IPA model '{model_name}' ({class_name}). Check config 'model_params' (especially required args like 'custom_pos_vocab_size') and the model's __init__ signature: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"CRITICAL: An unexpected error occurred initializing IPA model '{model_name}': {e}", exc_info=True)
            sys.exit(1)

    elif model_name in standard_models:
        model_class = standard_models[model_name]
        logger.info(f"Identified as standard model. Attempting to load class: {model_class.__name__}")
        try:
            init_args = {
                'vocab_size': vocab_size,
                'num_classes': num_classes,
                **model_params
            }
            sig = inspect.signature(model_class.__init__)
            valid_args = {k: v for k, v in init_args.items() if k in sig.parameters}
            invalid_args = {k: v for k, v in init_args.items() if k not in sig.parameters and k != 'self'}
            if invalid_args:
                logger.warning(f"Ignoring parameters not accepted by {model_class.__name__}.__init__: {invalid_args}")
            logger.info(f"Initializing {model_class.__name__} with parameters: {valid_args}")
            model = model_class(**valid_args)
        except TypeError as e:
            logger.error(f"TypeError initializing standard model '{model_name}' ({model_class.__name__}). Check config 'model_params' and __init__ signature: {e}")
            try:
                logger.warning("Attempting basic initialization (vocab_size, num_classes only)...")
                model = model_class(vocab_size=vocab_size, num_classes=num_classes)
            except Exception as basic_e:
                logger.error(f"CRITICAL: Basic initialization also failed for {model_name}: {basic_e}", exc_info=True)
                sys.exit(1)
        except Exception as e:
            logger.error(f"CRITICAL: An unexpected error occurred initializing standard model '{model_name}': {e}", exc_info=True)
            sys.exit(1)
    else:
        available_standard = list(standard_models.keys())
        available_ipa = [f"{k}_ipa" for k in ["transformer", "bilstm", "bigru", "textcnn"]]
        logger.error(f"CRITICAL: Unknown model name '{model_name}' provided in config.")
        logger.error(f"Available standard models are: {available_standard}")
        logger.error(f"Available IPA models (convention in config) are: {available_ipa}")
        sys.exit(1)

    if model is None:
        logger.error(f"CRITICAL: Model object is None after attempting to create '{model_name}'. Initialization failed unexpectedly.")
        sys.exit(1)

    logger.info(f"Model '{model_name}' (Class: {type(model).__name__}) created successfully.")
    return model.to(device)
