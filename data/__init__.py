from .data_IFP import build_loader_IFP


def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_IFP(config, logger)

