def create_model(args, logger=None):
    model = None

    if args.model_type == 'bi-causal':
        from exp.BiCausal_strategy2 import bicausal
        model = bicausal(args)
    else:
        raise ValueError("Model {} not recognized.".format(
            args.model_type))

    if logger:
        logger.info("--> model {} was created".format(
            args.model_type))

    return model
