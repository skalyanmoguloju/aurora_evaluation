from src.base.model import create_model as create_base_model

if __package__:
    from .layer import AuroRALinear
else:
    from layer import AuroRALinear


def create_model(rank, alpha, has_trainable_spline=False):
    model = create_base_model()

    for parameter in model.roberta.parameters():
        parameter.requires_grad = False


    for layer in model.roberta.encoder.layer:
        layer.attention.self.query = AuroRALinear(
            base_layer=layer.attention.self.query,
            rank=rank,
            alpha=alpha,
            has_trainable_spline=has_trainable_spline
        )

        layer.attention.self.value = AuroRALinear(
            base_layer=layer.attention.self.value,
            rank=rank,
            alpha=alpha,
            has_trainable_spline=has_trainable_spline
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_percentage = (
        trainable_parameters / total_parameters
    ) * 100

    print("Trainable parameters:", trainable_parameters)
    print("Total parameters:", total_parameters)
    print("Trainable percentage:", trainable_percentage)

    return model