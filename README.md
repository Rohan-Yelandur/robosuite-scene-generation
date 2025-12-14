# Project name TBD

A modular framework that determines failure modes in a robot policy and generates scenes to efficiently fine-tune.

[Link to the PDF Document](https://github.com/Rohan-Yelandur/robosuite-scene-generation/blob/b012bbc3be02971b38c747baed3691eb29e31751/Final_Paper.pdf)

## RoboMD Process
Works better for image agents vs low-dim agents b/c visual perturbations (colors, lighting) are meaningless to low-dim. However, it still kind of works since physical perturbations(cylinder size, table size) have meaning. 

train_discrete -> train_embedding -> train_continuous -> analyze_failures

Specific action descriptions found in action_dicts.py

Action XML modification found in latent_action_env.py

For any non-natively supported task (like transport), we'd have to define action_dicts, a custom environment class, xml modification logic, and register it in train_discrete.

## Credits
- [RoboMD](https://somsagar07.github.io/RoboMD/) for failure diagnosis.
- [Robomimic](https://robomimic.github.io/) for agent frameworks.
- [Robosuite](https://robosuite.ai/) for simulations.
