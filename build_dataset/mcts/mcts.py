import math
import numpy as np
import random
from env import EvolvingGraphEnv
import json
from utils import sent2action
from tqdm import tqdm
from evolving_graph.environment import EnvironmentGraph, EnvironmentState
import re
from collections import defaultdict
import cProfile
import pstats
import io
from pstats import SortKey
import time
import multiprocessing
from multiprocessing import Pool
import fire
from functools import partial

from contextlib import redirect_stdout

verb_to_idx = dict()
idx_to_verb = dict()

object_to_idx = dict()
idx_to_object = dict()


def get_id_from_object(obj):
    if obj not in object_to_idx.keys():
        new_id = len(object_to_idx)
        object_to_idx[obj] = new_id
        idx_to_object[new_id] = obj
        return new_id
    else:
        return object_to_idx[obj]


def get_id_from_verb(verb):
    if verb not in verb_to_idx.keys():
        new_id = len(verb_to_idx)
        verb_to_idx[verb] = new_id
        idx_to_verb[new_id] = verb
        return new_id
    else:
        return verb_to_idx[verb]


class Action:
    def __init__(self, verb: int, object_list: list[int], id_list: list[int]):
        self.verb = verb
        self.object_list = object_list
        self.id_list = id_list

    @staticmethod
    def from_str(s):
        verb = re.findall('\[(.*)\]', s)[0]
        objects = re.findall('<([a-z\_\-]+)>', s)
        ids = re.findall('\((\d*)\)', s)

        verb = get_id_from_verb(verb)
        objects = list(map(get_id_from_object, objects))

        ret = Action(verb, objects, ids)
        assert s == repr(ret)
        return ret

    def __repr__(self):
        s = f'[{idx_to_verb[self.verb]}]'
        for o, i in zip(self.object_list, self.id_list):
            s += f' <{idx_to_object[o]}> ({i})'
        return s


def get_action_list():
    '''
    action_list: all possible Actions from 'action_list.json'

    base_action_list:   list of actions, which requires 0 object OR with verb is "Walk"

    action_dict:        key is the object(s), value is the list of possible
    actions containing the given actions
    '''

    with open('action_list.json', 'r') as f:
        action_list = json.load(f)
    action_list = [sent2action(sent) for sent in action_list]

    # remove actions like "[Run] <bag> (1)" or "[Find] <alcohol> (1)"
    # removes 810 actions

    action_list = [action for action in action_list if 'Run' not in action and 'Find' not in action]

    action_list = list(map(Action.from_str, action_list))

    base_action_list = []
    action_dict = defaultdict(list)

    useless_actions = ['Drink', 'LookAt', 'PointAt', 'Greet', 'Read', 'Touch', 'Type', 'Watch', 'Push', 
                    'Pull', 'Sleep', 'WakeUp', 'Squeeze', 'Eat', 'Scrub', 'Cut', 'Wash']
    useless_actions = [get_id_from_verb(action) for action in useless_actions]
    for action in action_list:
        if action.object_list and action.object_list[0] in [get_id_from_object(obj) for obj in ['floor', 'wall', 'ceiling']]:
            continue
        elif action.verb in useless_actions:
            continue
        elif len(action.object_list) == 0 or action.verb == get_id_from_verb('Walk'):
            # e.g., [Sleep]
            # e.g., '[Walk] <address_book>'
            base_action_list.append(action)
        elif action.verb in [get_id_from_verb('Run'), get_id_from_verb('Find')]:
            continue
        elif len(action.object_list) == 1:
            # e.g., '[Close] <address_book>'
            action_dict[action.object_list[0]].append(action)
        elif len(action.object_list) == 2:
            # e.g., '[Pour] <after_shave> <arms_both>'
            objects_key = f'{action.object_list[0]}+{action.object_list[1]}'
            action_dict[objects_key].append(action)
    return action_list, base_action_list, action_dict


action_list, base_action_list, action_dict = get_action_list()


class State:
    def __init__(self, goals, unachieved_goals, useful_object_set, is_root=False, vh_state=None, dict_state=None, reward=0., action=None):
        self.env = EvolvingGraphEnv()
        if dict_state and not vh_state:
            vh_state = EnvironmentGraph(dict_state)
            vh_state = EnvironmentState(vh_state, self.env.name_equivalence, instance_selection=True)
        self.vh_state = vh_state

        self.untried_action_list = self.env.get_available_action_list2(
            vh_state, base_action_list, action_dict, useful_object_set, is_root=is_root
        )

        self.useful_object_set = useful_object_set
        self.goals = goals
        self.unachieved_goals = unachieved_goals
        self.reward = reward
        self.action = action

    def terminal(self):
        return len(self.unachieved_goals) == 0

    def next_state(self, is_random=False):
        list_len = len(self.untried_action_list)
        while list_len:  # some possible action has not been expanded, try each until some action succeed
            if is_random:
                index = random.randrange(list_len)
                action = self.untried_action_list[index]

                # swap the chosen and the last one
                self.untried_action_list[index], self.untried_action_list[list_len - 1] = self.untried_action_list[
                                                                                              list_len - 1], \
                                                                                          self.untried_action_list[
                                                                                              index]
            else:
                # choose any untried
                action = self.untried_action_list.pop()
            list_len -= 1
            reward, new_vh_state, unachieved_goals, success, modified_action = self.env.transition(action, self.vh_state, self.goals, self.unachieved_goals)
            if success:
                new_state = State(
                    self.goals, unachieved_goals, self.useful_object_set, vh_state=new_vh_state, reward=reward, action=modified_action
                )
                return new_state
            # do not need this since this func is only called once if is random
            '''
            else:
                if is_random:
                    """we swapped so that the failed action is in the end of the list, now remove it"""
                    assert action == self.untried_action_list[-1]
                    self.untried_action_list.pop()
                # print('fail', action)
            '''


class Node:
    def __init__(self, state, depth, parent=None):
        self.state = state
        self.depth = depth
        self.num_visited = 0
        self.quality_val = 0
        self.children = []
        self.parent = parent
        self.depth = depth

    def is_fully_expanded(self):
        return len(self.state.untried_action_list) == 0


use_profile = False


class MCTS:
    def __init__(self, num_simulation=10000, max_rollout_step=30, discount=0.9, c_init=10, c_base=1000):
        self.num_simulation = num_simulation
        self.max_rollout_step = max_rollout_step
        self.discount = discount
        self.c_init = c_init
        self.c_base = c_base

    def run(self, init_state, goals, cond_dict):
        def get_useful_object_set(cond_dict):
            room_list = ['home_office', 'bathroom', 'dining_room', 'bedroom']
            object_set = set(room_list)
            for node in cond_dict['nodes']:
                object_set.add(node.split(' || ')[0])
            for edge in cond_dict['edges']:
                splited_edge = edge.split(' || ')
                object_set.add(splited_edge[0])
                object_set.add(splited_edge[2])
            return object_set
        useful_object_set = get_useful_object_set(cond_dict)
        root_node = Node(State(goals, goals, useful_object_set, dict_state=init_state, is_root=True), depth=0)
        actions = []

        if use_profile:
            # Creating profile object
            ob = cProfile.Profile()
            ob.enable()

            path = self.search(root_node)

            ob.disable()
            sec = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
            ps.print_stats()
            print(sec.getvalue())

        else:
            path = self.search(root_node)
        return path

        '''
        action_len = i = 0
        while len(root_node.children):
            for node in root_node.children:
                print(
                    f'action: {node.state.action}, reward: {node.state.reward}, quality_val: {node.quality_val / node.num_visited}, num_visited: {node.num_visited}'
                )
            print()
            root_node = self.best_child(root_node, False)
            actions.append(root_node.state.action)
            i += 1
            if root_node.state.reward > 0:
                action_len = i
        actions = actions[:action_len] # remove wrong actions at the end of list
        return actions
        '''

    def best_child(self, node, is_explore=True):
        if is_explore:
            scores = []
            for child in node.children:
                exploration_rate = np.log((1 + node.num_visited + self.c_base) / self.c_base + self.c_init)
                u_score = exploration_rate * np.sqrt(
                    node.num_visited) / float(1 + child.num_visited)
                q_score = child.quality_val / child.num_visited
                scores.append(u_score + q_score)
        else:
            scores = [child.num_visited for child in node.children]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selected_child_index = random.choice(maxIndex)
        selected_child = node.children[selected_child_index]
        return selected_child

    def expand(self, node):
        if node.depth >= self.max_rollout_step:
            return node
        next_state = node.state.next_state()
        if next_state:
            new_node = Node(next_state, depth=node.depth + 1, parent=node)
            node.children.append(new_node)
            return new_node
        else:
            return node

    def tree_policy(self, node):
        while not node.state.terminal():
            if node.is_fully_expanded():
                node = self.best_child(node)
            else:
                return self.expand(node)
        return node

    def default_policy(self, node):
        discount = self.discount
        state = node.state
        reward = 0
        path = []
        for i in range(self.max_rollout_step - node.depth):
            if state.terminal():
                break
            state = state.next_state(is_random=True)
            if state:
                path.append((state.action, state.reward))
                reward += discount * state.reward
                discount *= discount
            else:
                break
        return reward, path

    def backup(self, node, reward, path):
        rev_path = []
        while node:
            if node.state.action:
                rev_path.append((node.state.action, node.state.reward))
            node.num_visited += 1
            reward += node.state.reward
            node.quality_val += reward
            reward *= self.discount
            node = node.parent
        return rev_path[::-1] + path

    def highest_reward_prefix(self, path):
        actions = [item[0] for item in path]
        rewards = [item[1] for item in path]
        mx_reward = cul_reward = 0
        mx_i = -1
        for i, reward in enumerate(rewards):
            cul_reward += self.discount * reward
            if cul_reward > mx_reward:
                mx_reward = cul_reward
                mx_i = i
        return mx_reward, actions[:mx_i+1]

    def search(self, root_node):
        cur = multiprocessing.current_process()
        if cur.name == 'MainProcess':
            # not using Process pool
            pidx = 0
        else:
            pidx = cur._identity[0]
        if pidx <= 1:
            progress = tqdm(range(self.num_simulation))
        else:
            progress = range(self.num_simulation)
        mx_reward, op_path = 0, []
        for _ in progress:
            expand_node = self.tree_policy(root_node)
            reward, path = self.default_policy(expand_node)
            path = self.backup(expand_node, reward, path)
            reward, path = self.highest_reward_prefix(path)
            if reward > mx_reward or (reward == mx_reward and len(path) < len(op_path)):
                # print(f'reward: {reward}, path: {path}')
                mx_reward = reward
                op_path = path

            best_child = max(root_node.children, key=lambda c: c.quality_val / c.num_visited)

            if pidx <= 1:
                progress.set_description(
                    f"score={root_node.quality_val / root_node.num_visited:.2f}, child_cnt={len(root_node.children)}, best_child_score={best_child.quality_val / best_child.num_visited:.2f}")

        if pidx <= 1:
            progress.close()
        '''
        for node in root_node.children:
            print(
                f'action: {node.state.action}, reward: {node.state.reward}, quality_val: {node.quality_val / node.num_visited}, num_visited: {node.num_visited}')
        '''
        output = []
        for action in op_path:
            output += action
        return output



def worker_func(x, out_dir):
    idx, data = x
    print('start:', idx)
    with open(os.path.join(out_dir, f'{idx}.json'), 'w') as g:
        # with redirect_stdout(f):
            sample = json.loads(data)
            # print(f'ready to run mcts for task: {sample["task"]}')
            # start_time = time.time()
            mcts = MCTS(num_simulation=20000, max_rollout_step=25, discount=0.99)
            graph_change = sample['graph_change']
            graph_change = set(graph_change["nodes"] + graph_change["edges"])
            # print('Goal:', graph_change)
            actions = mcts.run(sample['init_graph'], graph_change, sample['cond_dict'])
            sample['mcts_plan'] = [repr(act) for act in actions]
            g.write(json.dumps(sample) + '\n')
            # print('Plan: ', actions)
            # print('Ground truth plan: ', sample['demo'])
            # elapsed = time.time() - start_time
            # print(f'time used: {elapsed}')
            

import os


def main(input_file="output/train.json", out_dir="output/mcts"):
    os.makedirs(out_dir, exist_ok=True)
    print(f'loading: {input_file}')
    with open(input_file, 'r') as f:
        lines = f.readlines()

    print(f'loaded: {input_file}')
    print('total sampels:', len(lines))
    print('CPU count: ', multiprocessing.cpu_count())

    # use_cores = 1
    use_cores = max(multiprocessing.cpu_count()-1, 1) # leave one idle core

    if use_cores > 1:
        with Pool(processes=use_cores) as pool:
            pool.map(partial(worker_func, out_dir=out_dir), zip(range(len(lines)), lines))
    else:
        # development test only
        worker_func((0, lines[0]))


if __name__ == '__main__':
    fire.Fire(main)
