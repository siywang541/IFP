from .swin_transformer import build_swin
from .IFP import build_IFP


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_IFP(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
