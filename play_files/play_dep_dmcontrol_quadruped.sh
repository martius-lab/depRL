python -m deprl.play\
	--header 'import deprl, gym'\
	--env "deprl.environments.ControlSuite('quadruped-run', scaled_actions=False)" \
	--agent 'deprl.custom_agents.dep_factory(4, deprl.torch.agents.MPO())(replay=deprl.replays.buffers.Buffer(return_steps=3, batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=1000e4))' \
	--seed 0
