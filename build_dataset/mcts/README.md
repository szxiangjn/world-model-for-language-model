# Goal-Oriented Exploration with MCTS
Download RobotHow from [here](http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip) and unzip it.

`python build_dataset.py` to extract the dataset from RobotHow.

`python mcts.py` to do goal-oriented exploration with MCTS and generate embodied experience data.

Finally, the generated files are separate, so we want to merge them together by `python merge.py`.
