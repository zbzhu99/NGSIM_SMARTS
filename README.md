# How to run

1. Install SMARTS following the instructions in `./SMARTS` folder

2. Download the original data from <https://jbox.sjtu.edu.cn/v/link/view/75a0931a222347e1ba2e0441407f4a1f> and place it under `./ngsim` folder

3. Build NGSIM scenario with `scl scenario build --clean ./ngsim` (This will take a while)

4. Generate expert demonstrations with `python example_expert_generation.py`

5. Test rollout with `python example_rollout.py`


# Troubleshooting

1. **TypeError: export_glb() got an unexpected keyword argument 'extras'**
  
    Try install an alternative version of trimesh with:
    ```bash
    pip install trimesh==3.9.20
    ```
