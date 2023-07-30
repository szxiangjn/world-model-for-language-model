import json
import os
import argparse
from collections import defaultdict
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="programs_processed_precond_nograb_morepreconds", 
        help="path to RobotHow data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="path to output directory"
    )
    args = parser.parse_args()
    return args

def trans_graph(graph, node_specific=True):
    output = {}
    node_dict = {}
    for node in graph["nodes"]:
        node_dict[node["id"]] = {"class_name": node["class_name"], "states": set(node["states"])}
    if node_specific:
        nodes = node_dict
    else:
        nodes = set()
        for node in graph["nodes"]:
            name = node['class_name']
            for state in node['states']:
                nodes.add(f'{name} || {state}')
    output["nodes"] = nodes
    edges = set()
    for edge in graph["edges"]:
        edges.add(" || ".join([node_dict[edge['from_id']]['class_name'], edge['relation_type'], node_dict[edge['to_id']]['class_name']]))
    output["edges"] = edges
    return output


def find_graph_change(init_graph, final_graph):
    graph_change = {}
    init_graph = trans_graph(init_graph)
    final_graph = trans_graph(final_graph)

    node_change = set()
    for id in init_graph["nodes"]:
        class_name = init_graph["nodes"][id]["class_name"]
        init_state = init_graph["nodes"][id]["states"]
        final_state = final_graph["nodes"][id]["states"]
        state_change = final_state - init_state & final_state
        for state in state_change:
            node_change.add(f"{class_name} || {state}")
    graph_change["nodes"] = list(node_change)

    dup_edge_change = final_graph["edges"] - init_graph["edges"].intersection(final_graph["edges"]) # contains duplicate edges like A_rel_B and B_rel_A
    edge_change = set()
    for edge in dup_edge_change:
        node_1, rel, node_2 = edge.split(" || ")
        if rel == 'CLOSE':
            node_pair_list = [node_1, node_2]
            node_pair_list.sort()
            edge = " || ".join([node_pair_list[0], rel, node_pair_list[1]])
        edge_change.add(edge)

    graph_change["edges"] = list(edge_change)

    return graph_change

def rel2str(k, v):
    if k in ['free', 'atreach']:
        return None
    if k == 'location':
        return 'the ' + v[0][0] + ' is in the ' + v[1][0]
    if k == ['inside', 'in']:
        return 'the ' + v[0][0] + ' is ' + k + ' the ' + v[1][0]
    if k == 'is_off':
        return 'the ' + v[0] + ' is off'
    if k == 'is_on':
        return 'the ' + v[0] + ' is on'
    if k == 'plugged':
        return 'the ' + v[0] + ' is plugged'
    if k == 'closed':
        return 'the ' + v[0] + ' is closed'
    if k == 'facing':
        return 'the ' + v[0][0] + ' is facing the ' + v[1][0]
    else:
        return None

def cond2str(conditions):
    cond_str = ''
    for condition in conditions:
        for k, v in condition.items():
            rel_str = rel2str(k, v)
            if rel_str is not None:
                rel_str = rel_str.capitalize() + '.'
                rel_str = rel_str.replace('_', ' ')
                cond_str += ' ' + rel_str
    return cond_str[1:]

def cond2state(conditions):
    nodes = set()
    edges = set()
    for condition in conditions:
        for k, v in condition.items():
            if k in ['location', 'in', 'inside']:
                edges.add(f'{v[0][0]} || INSIDE || {v[1][0]}')
            if k == 'is_off':
                nodes.add(f'{v[0]} || OFF')
            if k == 'is_on':
                nodes.add(f'{v[0]} || ON')
            if k == 'closed':
                nodes.add(f'{v[0]} || CLOSED')
    output = {'nodes': list(nodes), 'edges': list(edges)}
    return output

def is_state_match_cond(cond_state, init_graph):
    init_state = trans_graph(init_graph, node_specific=False)
    for node in cond_state['nodes']:
        if node not in init_state['nodes']:
            '''
            print(node)
            for n in init_state['nodes']:
                if node.split(' || ')[0] in n:
                    print(n)
            for edge in init_state['edges']:
                if node.split(' || ')[0] in edge and ' CLOSE ' not in edge:
                    print(edge)

            print('***************')
            '''
            return False
    for edge in cond_state['edges']:
        if edge not in init_state['edges']:
            '''
            print(edge)
            edge_split = edge.split(' || ')
            obj1, obj2 = edge_split[0], edge_split[2]
            for n in init_state['nodes']:
                if obj1 in n or obj2 in n:
                    print(n)
            for edge in init_state['edges']:
                if obj1 in edge or obj2 in edge and ' CLOSE ' not in edge:
                    print(edge)

            print('***************')
            '''
            return False

    return True

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    data_dict = defaultdict(dict)

    # Get env_id and graphs
    scene_prefix = os.path.join(args.input_dir, "init_and_final_graphs")
    scenes = os.listdir(scene_prefix)
    for scene in scenes:
        print(f"\n\nStart {scene}...")
        env_id = int(scene[len("TrimmedTestScene")])
        dir_prefix = os.path.join(scene_prefix, scene)
        dirs = os.listdir(dir_prefix)
        for dir in dirs:
            print(dir)
            file_prefix = os.path.join(dir_prefix, dir)
            files = os.listdir(file_prefix)
            for file in tqdm(files):
                file_name = file.split(".")[0]
                key = os.path.join(dir, f"{file_name}_{env_id}")
                data_dict[key]["env_id"] = env_id
                with open(os.path.join(file_prefix, file), "r") as f:
                    states = json.load(f)
                    data_dict[key]["init_graph"] = states["init_graph"]
                    data_dict[key]["graph_change"] = find_graph_change(states["init_graph"], states["final_graph"])

    # Get task name
    dir_prefix = os.path.join(args.input_dir, "withoutconds")
    dirs = os.listdir(dir_prefix)
    for dir in dirs:
        file_prefix = os.path.join(dir_prefix, dir)
        files = os.listdir(file_prefix)
        for file in files:
            file_name = file.split(".")[0]
            with open(os.path.join(file_prefix, file), "r") as f:
                demo = []
                for i, line in enumerate(f):
                    if i == 0:
                        task = line.strip().lower()
                    elif i > 3:
                        if not line.startswith('[Find]'):
                            demo.append(line.strip())
            with open(os.path.join(file_prefix.replace('withoutconds', 'initstate'), file.replace('txt', 'json')), 'r') as f:
                conditions = json.load(f)
                cond_str = cond2str(conditions)
            # one program may be executed in different env
            for env_id in range(8):
                key = os.path.join(dir, f"{file_name}_{env_id}")
                if key in data_dict:
                    data_dict[key]["task"] = task
                    data_dict[key]['cond'] = cond_str
                    data_dict[key]['cond_dict'] = cond2state(conditions)
                    data_dict[key]['demo'] = demo
                    if not is_state_match_cond(data_dict[key]['cond_dict'], data_dict[key]['init_graph']):
                        data_dict.pop(key)
    
    dataset = list(data_dict.values())
    # check validity
    for key, sample in data_dict.items():
        assert "task" in sample
        assert "env_id" in sample
        assert "init_graph" in sample
        assert "graph_change" in sample
        assert sample["graph_change"]["nodes"] or sample["graph_change"]["edges"]

    random.shuffle(dataset)
    splits = {}
    splits["train"] = dataset[:int(0.7 * len(dataset))]
    splits["val"] = dataset[int(0.7 * len(dataset)): int(0.8 * len(dataset))]
    splits["test"] = dataset[int(0.8 * len(dataset)):]

    for split, data in splits.items():
        with open(os.path.join(args.output_dir, f"{split}.json"), "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    main(parse_args())
