# Height_CrowdNav

- Instructions for running DRL-VO and ORCA
  - DRL-VO: In `crowd_nav/configs/config.py`, set `robot.policy='om_gru'` and `env.env_name = 'CrowdSim3DTbObsHieOM-v0'`
  - ORCA: In `crowd_nav/configs/config.py`, set `env.env_name = 'CrowdSim3DTbObs-v0'`. In `test.py`, set `--orca` to `True`.
- Instructions for deploying a ClearPath Jackal robot in Atrium and Outdoor environments
  - Please adjust the routes of humans by modifying `configs/human_regions_shifted.csv` and adjust the layouts of obstacles by modifying `configs/obstacle_rectangles_shifted.csv`, based on your own real-world environment
  - In `crowd_nav/configs/config.py`, set `env.csl_workspace_type = 'gdc'`
  - In testing, set `env.env_name = 'rosTurtlebot2iEnv-v0'` in `config.py` in your trained checkpoint folder 
  - Hardware requirements: Only a zed2i and a Jackal are needed. Please check Appendix D for details on the robot setup.



