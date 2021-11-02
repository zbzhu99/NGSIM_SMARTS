# How to run

1. Install SMARTS following the instructions in `./SMARTS` folder

2. Download the original data from <https://jbox.sjtu.edu.cn/v/link/view/75a0931a222347e1ba2e0441407f4a1f> and place it under `./ngsim` folder

3. Build NGSIM scenario with `scl build --clean ./ngsim`

4. Generate expert demonstrations with `python example_expert_generation.py`

5. Test rollout with `python example_rollout.py`
