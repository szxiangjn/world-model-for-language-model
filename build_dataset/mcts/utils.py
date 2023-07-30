import os, re
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils

class Translator:
    def __init__(self, action_list_path, action_embeddings_path=None, model_id='stsb-roberta-large', batch_size=32, device=0):
        print('loading translator...')
        self.batch_size = batch_size
        self.device = device
        with open(action_list_path, "r") as f:
            self.action_list = json.load(f)
        self.action_list.append('\n')
        self.model = SentenceTransformer(model_id).to(self.device)
        if action_embeddings_path is not None and os.path.exists(action_embeddings_path):
            self.action_list_embedding = np.load(action_embeddings_path)
            self.action_list_embedding = torch.tensor(self.action_list_embedding).to(self.device)
        else:
            def process(action):
                action = action.replace('_', ' ')
                action = action[0].upper() + action[1:]
                return action
            processed_action_list = [process(action) for action in self.action_list]
            self.action_list_embedding = self.model.encode(processed_action_list, batch_size=batch_size, convert_to_tensor=True, device=self.device)
            if action_embeddings_path is not None:
                action_embeddings = self.action_list_embedding.cpu().numpy()
                np.save(action_embeddings_path, action_embeddings)
        
    def translate(self, query_batch):
        query_batch = [query.lower() for query in query_batch]
        query_batch_embedding = self.model.encode(query_batch, batch_size=self.batch_size, convert_to_tensor=True, device=self.device)
        cos_scores = st_utils.pytorch_cos_sim(query_batch_embedding, self.action_list_embedding).detach().cpu().numpy()
        translated_ids = np.argmax(cos_scores, axis=1)
        actions = [sent2action(self.action_list[idx]) for idx in translated_ids]
        sims = np.max(cos_scores, axis=1)
        return actions, sims

def sent2action(sent):
        if sent == '\n':
            return sent
        sent = sent.split()
        verb = sent[0]
        if verb in ['stand', 'wake']:
            verb = verb[0].upper() + verb[1:]
            action = f'[{verb}Up]'
        elif verb == 'sleep':
            action = '[Sleep]'
        elif verb in ['put', 'take']:
            if sent[1] == 'on':
                action = f'[PutOn] <{sent[-1]}> (1)'
            elif sent[1] == 'off':
                action = f'[PutOff] <{sent[-1]}> (1)'
            elif sent[1] == 'back':
                action = f'[PutObjBack] <{sent[-1]}> (1)'
            else:
                object_1 = sent[1]
                object_2 = sent[-1]
                prepos = sent[2]
                if prepos == 'on':
                    action = f'[PutBack] <{object_1}> (1) <{object_2}> (1)'
                else:
                    action = f'[PutIn] <{object_1}> (1) <{object_2}> (1)'
        elif verb == 'pour':
            object_1 = sent[1]
            object_2 = sent[-1]
            action = f'[Pour] <{object_1}> (1) <{object_2}> (1)'
        elif verb in ['switch', 'turn', 'look', 'plug', 'point']:
            verb = verb[0].upper() + verb[1:]
            prepos = sent[1]
            prepos = prepos[0].upper() + prepos[1:]
            action = f'[{verb}{prepos}] <{sent[-1]}> (1)'
        else:
            verb = verb[0].upper() + verb[1:]
            action = f'[{verb}] <{sent[-1]}> (1)'
        
        return action

def prog2sent(prog):
    action = re.findall('\[(.*)\]', prog)[0]
    objs = re.findall('<([a-z\_\-]+)>', prog)
    objs = [obj.replace('_', ' ') for obj in objs]

    for i in range(1, len(action)):
        c = action[i]
        if c.isupper():
            action = action[:i] + ' ' + action[i:].lower()
            break
        
    if action in ['Walk', 'Run']:
        sent = f'{action} to {objs[0]}'
    elif action in ['Sit', 'Lie', 'Type']:
        sent = f'{action} on {objs[0]}'
    elif not len(objs):
        sent = action
    elif action.startswith('Put'):
        if action in ['Put back', 'Put']:
            sent = f'Put {objs[0]} on {objs[1]}'
        elif action == 'Put in':
            sent = f'Put {objs[0]} in {objs[1]}'
        elif action in ['Put on', 'Put off']:
            sent = f'{action} {objs[0]}'
        elif action == 'Put objback':
            sent = f'Put back {objs[0]}'
    elif action == 'Pour':
        sent = f'Pour {objs[0]} into {objs[1]}'
    elif len(objs) == 1:
        sent = f'{action} {objs[0]}'

    sent += '.'
    return sent
