import itertools
import random as random_lib

import gym
import numpy as np
import torch
from gym.spaces import Dict, Box
from transformers import BertTokenizer

import virtualhome
# from unity_simulator.comm_unity import UnityCommunication
import virtualhome.simulation.evolving_graph.utils as utils
from evolving_graph.scripts import read_script_from_list_string
from evolving_graph.execution import ScriptExecutor, Relation
from evolving_graph.environment import *
from utils import Translator
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
import wandb
import copy
import re
from collections import defaultdict
from time import time


class EvolvingGraphEnv:
    def __init__(self, max_episode_length=200):
        self.max_episode_length = max_episode_length
        self.name_equivalence = utils.load_name_equivalence()

    def reset(self, init_graph, graph_change=None, demo=None):
        graph = EnvironmentGraph(init_graph)
        self.state = EnvironmentState(graph, self.name_equivalence, instance_selection=True)
        self.executor = ScriptExecutor(graph, self.name_equivalence)

        self.step_num = 0

        self.tgt_graph_change = set()
        if graph_change:
            self.tgt_graph_change = set(graph_change["nodes"] + graph_change["edges"])
        self.cur_graph_change = set()
        self.demo = set(demo) if demo else set()
        self.goal_num = len(self.tgt_graph_change)

    def filter_action_list(self, state, action_list):
        char = state.get_nodes_by_attr('class_name', 'character')[0]
        char_close_objs = state.get_nodes_from(char, Relation.CLOSE)
        char_close_obj_set = set()
        for obj in char_close_objs:
            char_close_obj_set.add(obj.class_name)

    def get_available_action_list(self, state, base_action_list, action_dict):
        char = state.get_nodes_by_attr('class_name', 'character')[0]
        close_to_char_set = set()
        action_list = list(base_action_list)
        for close_node in state.get_nodes_from(char, Relation.CLOSE):
            close_to_char_set.add(close_node.class_name)
        for obj in close_to_char_set:
            action_list += action_dict[obj]
        # return action_list

        new_action_list = []
        for action in action_list:
            should_add = True
            # script = read_script_from_list_string([action])
            verb = re.findall('\[(.*)\]', action)[0]
            objects = re.findall('<([a-z\_\-]+)>', action)
            if verb not in ['Walk', 'Find', 'Run']:  # should always add these actions
                for obj in objects:
                    if obj not in close_to_char_set:  # don't add objects that are not close
                        should_add = False
                        break

            if should_add:
                new_action_list.append(action)
        return new_action_list

    def get_available_action_list2(self, state, base_action_list, action_dict, useful_object_set, is_root=False):
        """a newer version"""
        import mcts
        action_list: list[mcts.Action] = list(base_action_list)

        # Only walk to a room in the first step
        if is_root:
            room_list = ['dining_room', 'home_office', 'bathroom', 'bedroom']
            room_list = [mcts.get_id_from_object(room) for room in room_list]
            action_list = [action for action in action_list if action.object_list and action.object_list[0] in room_list]
            random_lib.shuffle(action_list)
            return action_list

        char = state.get_nodes_by_attr('class_name', 'character')[0]
        close_to_char_set: set[int] = set()
        for close_node in state.get_nodes_from(char, Relation.CLOSE):
            close_to_char_set.add(mcts.get_id_from_object(close_node.class_name))
        for obj in close_to_char_set:
            action_list += action_dict[obj]
        # return action_list

        # add valid action with two objects
        for obj_1 in close_to_char_set:
            for obj_2 in close_to_char_set:
                object_key = f'{obj_1}+{obj_2}'
                action_list += action_dict[object_key]
        '''
        room_list = ['home_office', 'living_room', 'kitchen', 'bathroom', 'dining_room', 'bedroom']
        rel_obj_list = ['plate', 'table']
        action_list = [
            action for action in action_list if not action.object_list or mcts.idx_to_object[action.object_list[0]] in room_list + rel_obj_list
        ]
        '''
        final_action_list = []
        for action in action_list:
            if not action.object_list:
                final_action_list.append(action)
            else:
                action_object_set = set([mcts.idx_to_object[object_id] for object_id in action.object_list])
                if action_object_set.issubset(useful_object_set):
                    final_action_list.append(action)

        return final_action_list

    '''
    Find object before operating it
    '''

    def modify_action(self, state, action, add_preaction=True):
        import mcts
        if isinstance(action, str):
            action = mcts.Action.from_str(action)

        # convert verb and objects from int to string
        verb = mcts.idx_to_verb[action.verb]
        objects = list(map(lambda x: mcts.idx_to_object[x], action.object_list))

        # convert from object to string
        action = repr(action)

        if not objects:
            return read_script_from_list_string([action]), [action]

        pre_action = []

        char = state.get_nodes_by_attr('class_name', 'character')[0]
        close_to_char_dict = defaultdict(set)
        for close_node in state.get_nodes_from(char, Relation.CLOSE):
            close_to_char_dict[close_node.class_name].add(close_node.id)
        char_loc = state.get_nodes_from(char, Relation.INSIDE)[0].class_name
        for i, object in enumerate(objects):
            object_node_list = state.get_nodes_by_attr('class_name', object)
            if not object_node_list:
                return None, None
            # If seated then stand up
            if add_preaction and verb in ['Find', 'Walk', 'Run']:
                if State.SITTING in char.states:
                    pre_action.append('[StandUp]')
            if object in ['home_office', 'bathroom', 'dining_room', 'bedroom']:
                object_id = object_node_list[0].id
            else:
                # Check if the object is near the character
                if verb not in ['Find', 'Walk', 'Run']:
                    for object_node in object_node_list:
                        object_id = object_node.id
                        flag = 0
                        if object_id in close_to_char_dict[object]:
                            flag = 1
                            break
                    if not flag:
                        return None, None
                    # If the object is inside a container, open it
                    if add_preaction:
                        loc_nodes = state.get_nodes_from(object_node, Relation.INSIDE)
                        for loc_node in loc_nodes:
                                if loc_node.class_name not in ['home_office', 'bathroom', 'dining_room', 'bedroom'] \
                                    and NodeAttrIn(State.CLOSED, 'states').evaluate(loc_node):
                                    pre_action.append(f'[Open] <{loc_node.class_name}> ({loc_node.id})')

                # Check if the object is in the same room
                else:
                    object_id = None
                    object_loc_dict = {}
                    for object_node in object_node_list:
                        loc_nodes = state.get_nodes_from(object_node, Relation.INSIDE)
                        if char_loc in [loc_node.class_name for loc_node in loc_nodes]:
                            object_id = object_node.id
                            break
                    if not object_id:
                        return None, None
                    # If the object is inside a container, walk to the container instead
                    if add_preaction:
                        for loc_node in loc_nodes:
                            if loc_node.class_name not in ['home_office', 'bathroom', 'dining_room', 'bedroom'] \
                                and NodeAttrIn(State.CLOSED, 'states').evaluate(loc_node):
                                action = f'[Walk] <{loc_node.class_name}> (1)'
                                object_id = loc_node.id
                                break

            if len(objects) > 1 and i == 0:
                '''
                # check if the object is already on the hands
                objects_at_hand = state.get_nodes_from(char, Relation.HOLDS_LH) + state.get_nodes_from(char,
                                                                                                       Relation.HOLDS_RH)
                objects_at_hand = [node.id for node in objects_at_hand]
                if object_id not in objects_at_hand:
                    return None
                    pre_action.append(f'[Find] <{object}> ({object_id})')
                    pre_action.append(f'[Grab] <{object}> ({object_id})')
                '''
                action = action.replace(' (1) ', f' ({object_id}) ')
            else:
                '''
                if not verb in ['Find', 'Walk', 'Run']:
                    pre_action.append(f'[Find] <{object}> ({object_id})')
                '''
                action = action[:-2] + f'{object_id})'

        modified_action = pre_action + [action]
        script = read_script_from_list_string(modified_action)
        return script, modified_action

    def step(self, action, step_reward=2, step_penalty=0.1, out_demo_penalty=0.3):
        if action == '\n':
            return 0, True, {'success': True, 'change': set(),
                             'percentage': 1 - len(self.tgt_graph_change) / self.goal_num}
        action_out_demo = action not in self.demo
        script, _ = self.modify_action(self.state, action, add_preaction=True)
        if script:
            next_state = next(self.executor.find_solutions_rec(script, 0, self.state), None)
            success = next_state is not None
        else:
            success = False
        if success:
            self.state = next_state

        reward, change = self.reward(step_reward, step_penalty, action_out_demo * out_demo_penalty)
        self.step_num += 1
        done = False
        if self.step_num == self.max_episode_length:
            done = True
        percentage = 1 - len(self.tgt_graph_change) / self.goal_num
        info = {"success": success, 'change': change, 'percentage': percentage}

        return reward, done, info

    def trans_graph(self, state):
        nodes = set()
        for node_id, node in state._new_nodes.items():
            node_name = state.get_node(node_id).class_name
            for node_state in node.states:
                node_state = node_state.name
                nodes.add(' || '.join([node_name, node_state]))

        edges = set()
        for (from_node_id, rel), to_node_set in state._new_edges_from.items():
            from_node = state.get_node(from_node_id).class_name
            rel = rel.name
            for to_node_id in to_node_set:
                to_node = state.get_node(to_node_id).class_name
                edges.add(' || '.join([from_node, rel, to_node]))
        '''
                node_pair_list = [from_node, to_node]
                node_pair_list.sort()
                node_pair = ' || '.join(node_pair_list)
                if node_pair not in edges or rel != 'CLOSE':
                    if rel == 'CLOSE':
                        edge = ' || '.join([node_pair_list[0], 'CLOSE', node_pair_list[1]])
                    else:
                        edge = ' || '.join([from_node, rel, to_node])
                    edges[node_pair] = edge
        edges = {edge for edge in edges.values()}
        '''

        output = nodes | edges
        return output

    def reward(self, step_reward, step_penalty, out_demo_penalty=0):
        next_graph_change = self.trans_graph(self.state)
        one_step_change = next_graph_change - (next_graph_change & self.cur_graph_change)
        achieved_change = self.tgt_graph_change & one_step_change
        self.tgt_graph_change = self.tgt_graph_change - achieved_change
        self.cur_graph_change = next_graph_change
        return step_reward * int(len(achieved_change) > 0) - step_penalty - out_demo_penalty, one_step_change

    # repr(action) == '[Walk] <keyboard> (1)'

    def transition(self, action, state, goals, unachieved_goals):
        executor = ScriptExecutor(state._graph, self.name_equivalence)
        script, modified_action = self.modify_action(state, action)
        if script:
            next_state = next(executor.find_solutions_rec(script, 0, state), None)
            success = next_state is not None
        else:
            next_state = None
            success = False

        reward = new_goals = new_unachieved_goals = None
        if success:
            # TODO: should fix
            change = self.trans_graph(state)  # one step change of previous action
            next_change = self.trans_graph(next_state)  # one step change of current action
            change_all = (next_change & change)  # changes by both the last two steps
            one_step_change = next_change - change_all
            one_step_unchange = change - change_all

            # if (len(one_step_unchange) !=
            # 0):
            #     print('why')
            one_step_achieved_goals = unachieved_goals & one_step_change
            one_step_revoked_goals = goals & one_step_unchange
            new_unachieved_goals = unachieved_goals - one_step_achieved_goals
            reward = 2 * int(len(one_step_achieved_goals)) - 1.5 * int(len(one_step_revoked_goals)) - 0.1
            # reward = 2 * int(len(one_step_achieved_goals) > 0) - 0.5
            if next_change & goals == goals:
                reward += 5
            
            # print(one_step_revoked_goals)
        return reward, next_state, new_unachieved_goals, success, modified_action

    # next_state.get_nodes_from(next_state.get_nodes_by_attr('class_name', 'keyboard')[0], Relation.CLOSE)
    # next_state.get_nodes_from(next_state.get_nodes_by_attr('class_name', 'character')[0], Relation.CLOSE)

    def show_char_state(self):

        def print_node_names(n_list):
            print([n.class_name for n in n_list])

        state = self.state
        char = state.get_nodes_by_attr('class_name', 'character')[0]
        print("Character holds:")
        print_node_names(state.get_nodes_from(char, Relation.HOLDS_RH))
        print_node_names(state.get_nodes_from(char, Relation.HOLDS_LH))
        print("Character is on:")
        print_node_names(state.get_nodes_from(char, Relation.ON))
        print("Character is in:")
        print_node_names(state.get_nodes_from(char, Relation.INSIDE))


class EvolvingGraphGymEnv(gym.Env):
    def __init__(self, tokenizer, envs, prompt, max_episode_length=200, prompt_max_length=1000, action_max_length=10,
                 random=True, random_seed='EvolvingGraphGymEnv', rank=0, step=1):
        if random:
            self.random = random_lib.Random(f'{random_seed}{rank}')
            self.next_env = lambda: self.random.choice(envs)
        else:
            self.next_env = itertools.cycle(envs[i] for i in range(rank, len(envs), step)).__next__
        self.tokenizer: BertTokenizer = tokenizer
        self.prompt = prompt
        self.env = EvolvingGraphEnv(max_episode_length=max_episode_length)
        self.observation_space = gym.spaces.Box(low=-100, high=100000, shape=(2, prompt_max_length), dtype=np.int)
        self.action_space = gym.spaces.Box(low=-100, high=100000, shape=(2, action_max_length), dtype=np.int)
        self.prompt_max_length = prompt_max_length
        self.action_max_length = action_max_length
        self.log_table = None

    @staticmethod
    def join_input_ids(a, b):
        a_input = a['input_ids']
        a_att = a['attention_mask']
        b_raw_att = b['attention_mask']
        b_input = b['input_ids'][b_raw_att == 1]
        b_att = b_raw_att[b_raw_att == 1]
        return {'input_ids': torch.cat((a_input, b_input)), 'attention_mask': torch.cat((a_att, b_att))}

    def encode_tensor(self, tokenize_result):
        raw_length = tokenize_result['input_ids'].shape[0]
        if raw_length >= self.prompt_max_length:
            input_ids = tokenize_result['input_ids'][:self.prompt_max_length]
            attention_mask = tokenize_result['input_ids'][:self.prompt_max_length]
        else:
            # must padding to the left for generate to work
            input_ids = torch.nn.functional.pad(tokenize_result['input_ids'], (self.prompt_max_length - raw_length, 0),
                                                value=self.tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(tokenize_result['attention_mask'],
                                                     (self.prompt_max_length - raw_length, 0))
        return torch.stack((input_ids, attention_mask))

    def decode_tensor(self, tensor: np.ndarray):
        return {'input_ids': torch.tensor(tensor[0], device=self.device),
                'attention_mask': torch.tensor(tensor[1], device=self.device)}

    def reset(self):
        if self.log_table is not None:
            wandb.log({"episode trajectory": self.log_table})
        self.current_env = self.next_env()
        self.current_prompt = self.tokenizer(self.prompt % (self.current_env['task'], self.current_env['cond']),
                                             return_tensors='pt')
        self.current_prompt = {k: v[0].squeeze(0) for k, v in self.current_prompt.items()}
        self.device = self.current_prompt['input_ids'].device
        self.env.reset(self.current_env['init_graph'], self.current_env['graph_change'], self.current_env['demo'])
        self.log_table = wandb.Table(columns=['decoded', 'translated', 'success', 'reward'])
        self.log_table.add_data(self.current_env['task'], 'reset', True, 0.)
        # print(self.current_env['env_id'], self.current_env['task'])
        return self.encode_tensor(self.current_prompt)

    def step(self, action: tuple):
        input_ids, decoded, translated = action
        input_ids = {k: torch.tensor(t, device=self.current_prompt['input_ids'].device) for k, t in input_ids.items()}
        self.current_prompt = self.join_input_ids(self.current_prompt, input_ids)
        if translated == '\n':
            reward, done, info = 0., True, {'success': True}
        else:
            reward, done, info = self.env.step(translated)
        self.log_table.add_data(decoded, translated, info['success'], reward)
        return self.encode_tensor(self.current_prompt), reward, done, info

    def render(self, mode="human"):
        pass


class VecTranslator(VecEnvWrapper):
    def __init__(self, venv, tokenizer, action_list_path, device, action_embeddings_path):
        super().__init__(venv)
        self.tokenizer: BertTokenizer = tokenizer
        self.translator = Translator(action_list_path, batch_size=venv.num_envs,
                                     action_embeddings_path=action_embeddings_path)
        self.device = device

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def step_async(self, actions: np.ndarray) -> None:
        input_ids = actions[:, 0]
        attention_mask = actions[:, 1]
        # print(input_ids)
        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        translated, _ = self.translator.translate(decoded)
        self.venv.step_async(({'input_ids': input_ids[i], 'attention_mask': attention_mask[i]}, d, t) for i, (d, t) in
                             enumerate(zip(decoded, translated)))


def register_gym_envs():
    gym.register(id='EvolvingGraphGymEnv-v0', entry_point='env:EvolvingGraphGymEnv')


'''
class UnityEnv:
    def __init__(self, exec_path, max_episode_length=200, base_port=8080, port_id=0):
        self.max_episode_length = max_episode_length
        port_num = base_port + port_id
        self.comm = UnityCommunication(file_name=exec_path, port=str(port_num), x_display="0")

    def reset(self, env_id=1, init_graph=None):
        self.comm.reset(env_id - 1)
        if init_graph:
            success, _ = self.comm.expand_scene(init_graph)
            if not success:
                raise RuntimeError("Failed to initialize environment with given graph")
        else:
            self.comm.add_character()

        self.step_num = 0
        self.changed_graph = True

    def close(self):
        self.comm.close()

    def step(self, action):
        if isinstance(action, str):
            action = [action]
        success, _ = self.comm.render_script(script=action, find_solution=True, recording=False, skip_animation=True, image_synthesis=[])
        if success:
            self.changed_graph = True
        
        reward = self.reward()
        self.step_num += 1
        done = False
        if self.step_num == self.max_episode_length:
            done = True
        info = {"success": success}

        return None, reward, done, info

    def reward(self):
        # raise NotImplementedError
        return 0

    def get_graph(self):
        if self.changed_graph:
            _, graph = self.comm.environment_graph()
            self.changed_graph = False
        return graph
        '''
