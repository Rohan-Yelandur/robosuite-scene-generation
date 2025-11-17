# Project name TBD

A modular framework that determines failure modes in a robot policy and generates scenes to efficiently fine-tune.

## RoboMD Process
train_discrete -> train_embedding -> train_continuous -> analyze_failures

Specific action descriptions found in action_dicts.py

Action XML modification found in latent_action_env.py

## Credits
- [RoboMD](https://somsagar07.github.io/RoboMD/) for failure diagnosis.
- [Robomimic](https://robomimic.github.io/) for agent frameworks.
- [Robosuite](https://robosuite.ai/) for simulations.