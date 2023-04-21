import torch

from deprl import custom_torso
from deprl.vendor.tonic.torch import agents, updaters


class TunedMPO(agents.MPO):
    """Maximum a Posteriori Policy Optimisation.
    MPO: https://arxiv.org/pdf/1806.06920.pdf
    MO-MPO: https://arxiv.org/pdf/2005.07513.pdf
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_params(
        self,
        lr_critic=1e-3,
        grad_clip_critic=0,
        lr_actor=3e-4,
        lr_dual=1e-2,
        grad_clip_actor=0,
        hidden_size=None,
        batch_size=None,
        retnorm=None,
        return_steps=None,
    ):
        def optim_critic(params):
            return torch.optim.Adam(params, lr_critic)

        self.critic_updater = TunedExpectedSARSA(
            optimizer=optim_critic, gradient_clip=grad_clip_critic
        )

        def optim_actor(params):
            return torch.optim.Adam(params, lr=lr_actor)

        def optim_dual(params):
            return torch.optim.Adam(params, lr=lr_dual)

        self.actor_updater = TunedMaximumAPosteriori(
            actor_optimizer=optim_actor,
            dual_optimizer=optim_dual,
            gradient_clip=grad_clip_actor,
        )
        if hidden_size is None:
            hidden_size = 256
        if retnorm is not None:
            self.model = custom_torso.custom_return_mpo(
                hidden_size=hidden_size
            )
        else:
            self.model = custom_torso.custom_model_mpo(hidden_size=hidden_size)
        if batch_size is not None:
            self.replay.batch_size = batch_size
        if return_steps is not None:
            self.replay.return_steps = return_steps

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)


class TunedExpectedSARSA(updaters.critics.ExpectedSARSA):
    def __init__(
        self, num_samples=20, loss=None, optimizer=None, gradient_clip=0
    ):
        self.num_samples = num_samples
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
        )
        self.gradient_clip = gradient_clip


class TunedMaximumAPosteriori(
    updaters.actors.MaximumAPosterioriPolicyOptimization
):
    def __init__(
        self,
        num_samples=20,
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        epsilon_std=1e-6,
        initial_log_temperature=1.0,
        initial_log_alpha_mean=1.0,
        initial_log_alpha_std=10.0,
        min_log_dual=-18.0,
        per_dim_constraining=True,
        action_penalization=True,
        actor_optimizer=None,
        dual_optimizer=None,
        gradient_clip=0,
    ):
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.epsilon_mean = epsilon_mean
        self.epsilon_std = epsilon_std
        self.initial_log_temperature = initial_log_temperature
        self.initial_log_alpha_mean = initial_log_alpha_mean
        self.initial_log_alpha_std = initial_log_alpha_std
        self.min_log_dual = torch.as_tensor(min_log_dual, dtype=torch.float32)
        self.action_penalization = action_penalization
        self.epsilon_penalty = epsilon_penalty
        self.per_dim_constraining = per_dim_constraining
        self.actor_optimizer = actor_optimizer or (
            lambda params: torch.optim.Adam(params, lr=3e-4, weight_decay=1e-5)
        )
        self.dual_optimizer = dual_optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-2, weight_decay=1e-5)
        )
        self.gradient_clip = gradient_clip
