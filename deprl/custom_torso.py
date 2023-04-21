import torch

from deprl.vendor.tonic.torch import models, normalizers


def custom_model_mpo(hidden_size=256):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((hidden_size, hidden_size), torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((hidden_size, hidden_size), torch.nn.ReLU),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def custom_return_mpo(hidden_size=256):
    return models.ActorCriticWithTargets(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((hidden_size, hidden_size), torch.nn.ReLU),
            head=models.GaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP((hidden_size, hidden_size), torch.nn.ReLU),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
        return_normalizer=normalizers.returns.Return(0.99),
    )
