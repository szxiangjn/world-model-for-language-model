import os
import json

dir_path = 'output/mcts'
files = os.listdir(dir_path)
with open('train.json', 'w') as f:
    for file in files:
        with open(os.path.join(dir_path, file), 'r') as g:
            for line in g:
                sample = json.loads(line)
                sample['gt_demo'] = sample['demo']
                sample['demo'] = sample['mcts_plan']
                del sample['mcts_plan']
                f.write(json.dumps(sample) + '\n')
