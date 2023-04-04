python -m deprl.play\
	--header 'import tonic.torch; from deprl import custom_agents; custom_agents;import gym;import ostrichrl;'\
	--env "tonic.environments.ControlSuite('quadruped-run', scaled_actions=False)" \
	--agent 'custom_agents.dep_factory(4, tonic.torch.agents.MPO())(replay=tonic.replays.buffers.Buffer(return_steps=3, batch_size=256, steps_between_batches=1000, batch_iterations=30, steps_before_batches=1000e4))' \
	--seed 0
