# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import functools
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom

import common.layers as layers
from common.text.text_processing import TextProcessing
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu, trim_audio_silence

from itertools import chain
import random

from transformers import BertTokenizer
from collections import defaultdict

class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        #self.bank = functools.lru_cache(beta_binomial_prior_distribution)
        f = functools.lru_cache(maxsize=128)
        self.bank = f(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':

        snd, sr = librosa.load(wav)
        snd = trim_audio_silence(snd.astype(np.float32))
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0, wav + "/" + str(mel_len) + "/" + str(pitch_mel.shape[0])

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 symbol_set='english_phonimized',
                 p_arpabet=1.0,
                 n_speakers=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=False,
                 append_space_to_text=False,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 **ignored):

        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dataset_path = dataset_path
        self.audiopaths_and_text = load_filepaths_and_text(
            dataset_path, audiopaths_and_text,
            has_speakers=(n_speakers > 1))
        print(len(self.audiopaths_and_text))
        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.load_pitch_from_disk = load_pitch_from_disk

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')

        self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)
        self.space = [self.tp.encode_text("A A")[0][3]]
        self.n_speakers = n_speakers
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator

        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

        expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1))

        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        if len(self.audiopaths_and_text[0]) < expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
                             'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]')

        if len(self.audiopaths_and_text[0]) > expected_columns:
            print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)

        # NLP
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # dependency parser
        self.deprel_to_idx = self.get_vocab('filelists/deprel_list.txt')
        self.upos_to_idx = self.get_vocab('filelists/upos_list.txt')
        self.xpos_to_idx = self.get_vocab('filelists/xpos_list.txt')
        self.tree_relation_to_idx = {'[PAD]': 0, '[SELF]': 1, '[PAR]': 2, '[CHI]': 3}

    def get_vocab(self, path):
        vocab_to_idx = {'Recursive': 0}
        vocab_idx = 1
        f_vocab = open(path, 'r', encoding='utf8')
        while True:
            line = f_vocab.readline()
            if not line: break
            vocab_to_idx[line.strip()] = vocab_idx
            vocab_idx += 1
        f_vocab.close()
        return vocab_to_idx

    def __getitem__(self, index):
        # Separate filename and text
        if self.n_speakers > 1:
            audiopath, *extra, text, speaker = self.audiopaths_and_text[index]
            speaker = int(speaker)
        else:
            audiopath, *extra, text = self.audiopaths_and_text[index]
            speaker = None

        audiopath_split = audiopath.split('/')
        audiopath_split[2] = 'deps'
        audiopath_split[4] = audiopath_split[4][:-3] + 'json'

        # get dependency
        dep_path = '/'.join(audiopath_split)
        dep_text = None
        with open(dep_path, "r") as json_file:
            dep_text = json.load(json_file)
        raw_text = dep_text['text']

        dep_words = []
        for sent in dep_text['sentences']:
            dep_words.extend(sent['words'])

        # get phoneme input
        phoneme_input_text = ' '.join([word['text'] for sent in dep_text['sentences'] for word in sent['words']])

        # get tokenized text
        tokenized_text = self.tokenizer.tokenize(raw_text)
        input_tokens = [self.tokenizer.cls_token] + tokenized_text + [self.tokenizer.sep_token]
        segment_ids = [1 for i in range(len(tokenized_text) + 2)]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # get subword_to_word mapping
        subword_to_word_mapping_list = self.get_subword_to_word_mapping_list(dep_text, dep_words, tokenized_text.copy())

        # get neighbor
        forward_neighbor_list, backward_neighbor_list, forward_deprel_neighbor_list, backward_deprel_neighbor_list = self.get_neighbors(dep_text, subword_to_word_mapping_list)

        # get ancestor
        ancestor_list, ancestor_deprel_list, ancestor_xpos_list, prev_sen_len_list = self.get_ancestors(dep_text, dep_words, subword_to_word_mapping_list)

        # get next word relations
        next_word_relation_node_idx_list, next_word_relation_deprel_idx_list, next_word_relation_child_parent_list, next_word_relation_upos_list, next_word_relation_xpos_list, prev_word_relation_node_idx_list, prev_word_relation_deprel_idx_list, prev_word_relation_child_parent_list, prev_word_relation_upos_list, prev_word_relation_xpos_list = \
            self.get_adjacent_word_relations(dep_words, ancestor_list, prev_sen_len_list)

        if dep_text['num_words'] + 2 != len(subword_to_word_mapping_list):
            print('error')
            print(dep_text['text'])

        mel = self.get_mel(audiopath)
        text, word_level_lens = self.get_text(phoneme_input_text)
        text = torch.LongTensor(text)

        pitch = self.get_pitch(index, mel.size(-1))
        energy = torch.norm(mel.float(), dim=0, p=2)
        attn_prior = self.get_prior(index, mel.shape[1], text.shape[0])

        if pitch.size(-1) > mel.size(-1):
            pitch = pitch[:, :mel.size(-1)]
        elif pitch.size(-1) < mel.size(-1):
            mel = mel[:, :pitch.size(-1)]

        assert pitch.size(-1) == mel.size(-1)

        # No higher formants?
        if len(pitch.size()) == 1:
            pitch = pitch[None, :]


        nlp_data = dict()
        nlp_data['subword_ids'] = input_ids
        nlp_data['subword_segment_ids'] = segment_ids
        nlp_data['subword_to_word_mapping_ids'] = subword_to_word_mapping_list
        nlp_data['forward_neighbor_list'] = forward_neighbor_list
        nlp_data['backward_neighbor_list'] = backward_neighbor_list
        nlp_data['forward_deprel_neighbor_list'] = forward_deprel_neighbor_list
        nlp_data['backward_deprel_neighbor_list'] = backward_deprel_neighbor_list
        nlp_data['phoneme_to_word_mapping_list'] = word_level_lens
        nlp_data['ancestor_list'] = ancestor_list
        nlp_data['ancestor_deprel_list'] = ancestor_deprel_list
        nlp_data['ancestor_xpos_list'] = ancestor_xpos_list
        nlp_data['next_word_relation_node_idx_list'] = next_word_relation_node_idx_list
        nlp_data['next_word_relation_deprel_idx_list'] = next_word_relation_deprel_idx_list
        nlp_data['next_word_relation_child_parent_list'] = next_word_relation_child_parent_list
        nlp_data['next_word_relation_upos_list'] = next_word_relation_upos_list
        nlp_data['next_word_relation_xpos_list'] = next_word_relation_xpos_list
        nlp_data['prev_word_relation_node_idx_list'] = prev_word_relation_node_idx_list
        nlp_data['prev_word_relation_deprel_idx_list'] = prev_word_relation_deprel_idx_list
        nlp_data['prev_word_relation_child_parent_list'] = prev_word_relation_child_parent_list
        nlp_data['prev_word_relation_upos_list'] = prev_word_relation_upos_list
        nlp_data['prev_word_relation_xpos_list'] = prev_word_relation_xpos_list

        return (text, mel, len(text), pitch, energy, speaker, attn_prior,
                audiopath, nlp_data)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_subword_to_word_mapping_list(self, dep_text, dep_words, tokenized_text):
        tokenized_text_copy = [self.tokenizer.cls_token] + tokenized_text.copy()
        subword_to_word_mapping_list = [[0], []]  # CLS init
        cur_tok_idx = 1
        cur_tgt_word_idx = 0
        cur_tokenized_word = tokenized_text_copy[cur_tok_idx]
        cur_word = ''

        while cur_tgt_word_idx < dep_text['num_words']:
            if cur_word == dep_words[cur_tgt_word_idx]['text']:
                cur_tgt_word_idx += 1
                if cur_tgt_word_idx >= dep_text['num_words']:
                    break
                cur_word = ''
                subword_to_word_mapping_list.append([])
                continue
            else:
                if (len(cur_tokenized_word) > 2) and (cur_tokenized_word[:2] == '##'):
                    cur_tokenized_word = cur_tokenized_word[2:]
                if len(cur_tokenized_word) < len(dep_words[cur_tgt_word_idx]['text']):
                    if cur_tokenized_word == dep_words[cur_tgt_word_idx]['text'][
                                             len(cur_word):len(cur_word) + len(cur_tokenized_word)]:
                        subword_to_word_mapping_list[-1].append(cur_tok_idx)
                        cur_word += cur_tokenized_word
                        cur_tok_idx += 1
                        if cur_tok_idx >= len(tokenized_text_copy):
                            cur_tokenized_word = None
                            continue
                        else:
                            cur_tokenized_word = tokenized_text_copy[cur_tok_idx]
                elif len(cur_tokenized_word) > len(dep_words[cur_tgt_word_idx]['text']):
                    if cur_tokenized_word[:len(dep_words[cur_tgt_word_idx]['text'])] == dep_words[cur_tgt_word_idx]['text']:
                        cur_word = cur_tokenized_word[:len(dep_words[cur_tgt_word_idx]['text'])]
                        subword_to_word_mapping_list[-1].append(cur_tok_idx)
                        cur_tokenized_word = cur_tokenized_word[len(dep_words[cur_tgt_word_idx]['text']):]
                    else:
                        raise Exception(cur_tokenized_word+'\n'+dep_words[cur_tgt_word_idx]['text']+'\n'+' '.join(tokenized_text_copy))
                else:
                    if cur_tokenized_word == dep_words[cur_tgt_word_idx]['text']:
                        cur_word = cur_tokenized_word
                        subword_to_word_mapping_list[-1].append(cur_tok_idx)
                        cur_tok_idx += 1
                        if cur_tok_idx >= len(tokenized_text_copy):
                            cur_tokenized_word = None
                            continue
                        else:
                            cur_tokenized_word = tokenized_text_copy[cur_tok_idx]
                    else:
                        raise Exception(cur_tokenized_word + '\n' + dep_words[cur_tgt_word_idx]['text'] + '\n' + ' '.join(tokenized_text_copy))

        subword_to_word_mapping_list.append([cur_tok_idx])  # SEP
        return subword_to_word_mapping_list

    def get_neighbors(self, dep_text, subword_to_word_mapping_list):
        forward_neighbor_list = [[i] for i in range(len(subword_to_word_mapping_list))]
        backward_neighbor_list = [[i] for i in range(len(subword_to_word_mapping_list))]
        forward_deprel_neighbor_list = [[self.deprel_to_idx['Recursive']] for i in
                                        range(len(subword_to_word_mapping_list))]
        backward_deprel_neighbor_list = [[self.deprel_to_idx['Recursive']] for i in
                                         range(len(subword_to_word_mapping_list))]

        prev_sen_len = 0
        for sent in dep_text['sentences']:
            for word in sent['words']:
                if word['head'] != 0:
                    if not word['deprel'] in self.deprel_to_idx:
                        raise Exception(word['deprel'])

                    forward_neighbor_list[word['head'] + prev_sen_len].append(word['id'] + prev_sen_len)
                    forward_deprel_neighbor_list[word['head'] + prev_sen_len].append(self.deprel_to_idx[word['deprel']])
                    backward_neighbor_list[word['id'] + prev_sen_len].append(word['head'])
                    backward_deprel_neighbor_list[word['id'] + prev_sen_len].append(self.deprel_to_idx[word['deprel']])
            prev_sen_len += len(sent['words'])

        return forward_neighbor_list, backward_neighbor_list, forward_deprel_neighbor_list, backward_deprel_neighbor_list

    def get_ancestors(self, dep_text, dep_words, subword_to_word_mapping_list):
        ancestor_list = [[] for i in range(len(subword_to_word_mapping_list))]
        ancestor_deprel_list = [[] for i in range(len(subword_to_word_mapping_list))]
        ancestor_xpos_list = [[] for i in range(len(subword_to_word_mapping_list))]

        prev_sen_len = 0
        # if len(dep_text['sentences']) > 1:
        #     print()
        prev_sen_len_list = []
        for sent in dep_text['sentences']:
            for i in range(len(sent['words'])):
                if not dep_words[i + prev_sen_len]['deprel'] in self.deprel_to_idx:
                    raise Exception(dep_words[i + prev_sen_len]['deprel'])

                cur_node_id = dep_words[i + prev_sen_len]['id']
                while cur_node_id != 0:
                    ancestor_list[dep_words[i + prev_sen_len]['id'] + prev_sen_len].append(dep_words[cur_node_id - 1 + prev_sen_len]['id'] + prev_sen_len)
                    ancestor_deprel_list[dep_words[i + prev_sen_len]['id'] + prev_sen_len].append(self.deprel_to_idx[dep_words[cur_node_id - 1 + prev_sen_len]['deprel']])
                    ancestor_xpos_list[dep_words[i + prev_sen_len]['id'] + prev_sen_len].append(self.xpos_to_idx[dep_words[cur_node_id - 1 + prev_sen_len]['xpos']])
                    cur_node_id = dep_words[cur_node_id - 1 + prev_sen_len]['head']
            prev_sen_len += len(sent['words'])
            prev_sen_len_list.append(prev_sen_len)
        ancestor_list[0].append(0)
        ancestor_list[-1].append(len(ancestor_list) - 1)
        ancestor_deprel_list[0].append(0)
        ancestor_deprel_list[-1].append(0)
        ancestor_xpos_list[0].append(0)
        ancestor_xpos_list[-1].append(0)

        return ancestor_list, ancestor_deprel_list, ancestor_xpos_list, prev_sen_len_list

    def get_adjacent_word_relations(self, dep_words, ancestor_list, prev_sen_len_list):
        next_word_relation_node_idx_list = []
        next_word_relation_deprel_idx_list = []
        next_word_relation_child_parent_list = []
        next_word_relation_upos_list = []
        next_word_relation_xpos_list = []

        prev_word_relation_node_idx_list = []
        prev_word_relation_deprel_idx_list = []
        prev_word_relation_child_parent_list = []
        prev_word_relation_upos_list = []
        prev_word_relation_xpos_list = []

        for i in range(len(ancestor_list)):
            # check cls / sep
            if i == 0 or i == (len(ancestor_list) - 1):
                next_word_relation_node_idx_list.append([i])
                next_word_relation_deprel_idx_list.append([self.deprel_to_idx['Recursive']])
                next_word_relation_child_parent_list.append([self.tree_relation_to_idx['[SELF]']])
                next_word_relation_upos_list.append([self.upos_to_idx['Recursive']])
                next_word_relation_xpos_list.append([self.xpos_to_idx['Recursive']])

                prev_word_relation_node_idx_list.append([i])
                prev_word_relation_deprel_idx_list.append([self.deprel_to_idx['Recursive']])
                prev_word_relation_child_parent_list.append([self.tree_relation_to_idx['[SELF]']])
                prev_word_relation_upos_list.append([self.upos_to_idx['Recursive']])
                prev_word_relation_xpos_list.append([self.xpos_to_idx['Recursive']])
                continue

            cur_ancestors = ancestor_list[i]
            next_ancestors = ancestor_list[i + 1]
            prev_ancestors = ancestor_list[i - 1]

            cur_next_word_relation_node_idx_list = [cur_ancestors[0]]
            cur_next_word_relation_deprel_idx_list = [self.deprel_to_idx[dep_words[cur_ancestors[0] - 1]['deprel']]]
            cur_next_word_relation_child_parent_list = [self.tree_relation_to_idx['[SELF]']]
            cur_next_word_relation_upos_list = [self.upos_to_idx[dep_words[cur_ancestors[0] - 1]['upos']]]
            cur_next_word_relation_xpos_list = [self.xpos_to_idx[dep_words[cur_ancestors[0] - 1]['xpos']]]

            cur_prev_word_relation_node_idx_list = [cur_ancestors[0]]
            cur_prev_word_relation_deprel_idx_list = [self.deprel_to_idx[dep_words[cur_ancestors[0] - 1]['deprel']]]
            cur_prev_word_relation_child_parent_list = [self.tree_relation_to_idx['[SELF]']]
            cur_prev_word_relation_upos_list = [self.upos_to_idx[dep_words[cur_ancestors[0] - 1]['upos']]]
            cur_prev_word_relation_xpos_list = [self.xpos_to_idx[dep_words[cur_ancestors[0] - 1]['xpos']]]

            # ONLY next
            # last token or next sentence token
            if i == (len(ancestor_list) - 2) or (len(prev_sen_len_list) > 1 and prev_sen_len_list[:-1].__contains__(i)):
                # if len(prev_sen_len_list) > 1 and prev_sen_len_list[:-1].__contains__(i):
                #     print()
                next_word_relation_node_idx_list.append(cur_next_word_relation_node_idx_list)
                next_word_relation_deprel_idx_list.append(cur_next_word_relation_deprel_idx_list)
                next_word_relation_child_parent_list.append(cur_next_word_relation_child_parent_list)
                next_word_relation_upos_list.append(cur_next_word_relation_upos_list)
                next_word_relation_xpos_list.append(cur_next_word_relation_xpos_list)
            else:
                for j in range(len(cur_ancestors)):
                    if cur_ancestors[j] == next_ancestors[0]:
                        break
                    if not next_ancestors.__contains__(cur_ancestors[j]):
                        cur_next_word_relation_node_idx_list.append(cur_ancestors[j + 1])
                        cur_next_word_relation_deprel_idx_list.append(self.deprel_to_idx[dep_words[cur_ancestors[j + 1] - 1]['deprel']])
                        cur_next_word_relation_child_parent_list.append(self.tree_relation_to_idx['[PAR]'])
                        cur_next_word_relation_upos_list.append(self.upos_to_idx[dep_words[cur_ancestors[j + 1] - 1]['upos']])
                        cur_next_word_relation_xpos_list.append(self.xpos_to_idx[dep_words[cur_ancestors[j + 1] - 1]['xpos']])
                        continue
                    else:
                        std_idx = next_ancestors.index(cur_ancestors[j])
                        if std_idx == 0:
                            raise Exception('std_idx error')
                        for k in range(std_idx, 0, -1):
                            cur_next_word_relation_node_idx_list.append(next_ancestors[k - 1])
                            cur_next_word_relation_deprel_idx_list.append(self.deprel_to_idx[dep_words[next_ancestors[k - 1] - 1]['deprel']])
                            cur_next_word_relation_child_parent_list.append(self.tree_relation_to_idx['[CHI]'])
                            cur_next_word_relation_upos_list.append(self.upos_to_idx[dep_words[next_ancestors[k - 1] - 1]['upos']])
                            cur_next_word_relation_xpos_list.append(self.xpos_to_idx[dep_words[next_ancestors[k - 1] - 1]['xpos']])
                        break
                next_word_relation_node_idx_list.append(cur_next_word_relation_node_idx_list)
                next_word_relation_deprel_idx_list.append(cur_next_word_relation_deprel_idx_list)
                next_word_relation_child_parent_list.append(cur_next_word_relation_child_parent_list)
                next_word_relation_upos_list.append(cur_next_word_relation_upos_list)
                next_word_relation_xpos_list.append(cur_next_word_relation_xpos_list)

            # ONLY prev
            # first token or next_sentence_token start
            if i == 1 or (len(prev_sen_len_list) > 1 and prev_sen_len_list[:-1].__contains__(i-1)):
                # if len(prev_sen_len_list) > 1 and prev_sen_len_list[:-1].__contains__(i-1):
                #     print()
                prev_word_relation_node_idx_list.append(cur_prev_word_relation_node_idx_list)
                prev_word_relation_deprel_idx_list.append(cur_prev_word_relation_deprel_idx_list)
                prev_word_relation_child_parent_list.append(cur_prev_word_relation_child_parent_list)
                prev_word_relation_upos_list.append(cur_prev_word_relation_upos_list)
                prev_word_relation_xpos_list.append(cur_prev_word_relation_xpos_list)

            else:
                for j in range(len(cur_ancestors)):
                    if cur_ancestors[j] == prev_ancestors[0]:
                        break
                    if not prev_ancestors.__contains__(cur_ancestors[j]):
                        cur_prev_word_relation_node_idx_list.append(cur_ancestors[j + 1])
                        cur_prev_word_relation_deprel_idx_list.append(self.deprel_to_idx[dep_words[cur_ancestors[j + 1] - 1]['deprel']])
                        cur_prev_word_relation_child_parent_list.append(self.tree_relation_to_idx['[PAR]'])
                        cur_prev_word_relation_upos_list.append(self.upos_to_idx[dep_words[cur_ancestors[j + 1] - 1]['upos']])
                        cur_prev_word_relation_xpos_list.append(self.xpos_to_idx[dep_words[cur_ancestors[j + 1] - 1]['xpos']])
                        continue
                    else:
                        std_idx = prev_ancestors.index(cur_ancestors[j])
                        if std_idx == 0:
                            raise Exception('std_idx error')
                        for k in range(std_idx, 0, -1):
                            cur_prev_word_relation_node_idx_list.append(prev_ancestors[k - 1])
                            cur_prev_word_relation_deprel_idx_list.append(self.deprel_to_idx[dep_words[prev_ancestors[k - 1] - 1]['deprel']])
                            cur_prev_word_relation_child_parent_list.append(self.tree_relation_to_idx['[CHI]'])
                            cur_prev_word_relation_upos_list.append(self.upos_to_idx[dep_words[prev_ancestors[k - 1] - 1]['upos']])
                            cur_prev_word_relation_xpos_list.append(self.xpos_to_idx[dep_words[prev_ancestors[k - 1] - 1]['xpos']])
                        break
                prev_word_relation_node_idx_list.append(cur_prev_word_relation_node_idx_list)
                prev_word_relation_deprel_idx_list.append(cur_prev_word_relation_deprel_idx_list)
                prev_word_relation_child_parent_list.append(cur_prev_word_relation_child_parent_list)
                prev_word_relation_upos_list.append(cur_prev_word_relation_upos_list)
                prev_word_relation_xpos_list.append(cur_prev_word_relation_xpos_list)

        return next_word_relation_node_idx_list, next_word_relation_deprel_idx_list, next_word_relation_child_parent_list, next_word_relation_upos_list, next_word_relation_xpos_list,\
               prev_word_relation_node_idx_list, prev_word_relation_deprel_idx_list, prev_word_relation_child_parent_list, prev_word_relation_upos_list, prev_word_relation_xpos_list

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text, word_level_lens = self.tp.encode_text(text)

        if self.prepend_space_to_text:
            text = self.space + text

        if self.append_space_to_text:
            text = text + self.space

        return text, word_level_lens

    def get_prior(self, index, mel_len, text_len):

        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len,
                                                                   text_len))

        if self.betabinomial_tmp_dir is not None:
            audiopath, *_ = self.audiopaths_and_text[index]
            fname = Path(audiopath).relative_to(self.dataset_path)
            fname = fname.with_suffix('.pt')
            cached_fpath = Path(self.betabinomial_tmp_dir, fname)

            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)

        if self.betabinomial_tmp_dir is not None:
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_prior, cached_fpath)

        return attn_prior

    def get_pitch(self, index, mel_len=None):
        audiopath, *fields = self.audiopaths_and_text[index]

        if self.n_speakers > 1:
            spk = int(fields[-1])
        else:
            spk = 0

        if self.load_pitch_from_disk:
            pitchpath = fields[0]
            pitch = torch.load(pitchpath)
            if self.pitch_mean is not None:
                assert self.pitch_std is not None
                pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)
            return pitch

        if self.pitch_tmp_dir is not None:
            fname = Path(audiopath).relative_to(self.dataset_path)
            fname_method = fname.with_suffix('.pt')
            cached_fpath = Path(self.pitch_tmp_dir, fname_method)
            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        # No luck so far - calculate
        wav = audiopath
        if not wav.endswith('.wav'):
            wav = re.sub('/mels/', '/wavs/', wav)
            wav = re.sub('.pt$', '.wav', wav)

        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method,
                                   self.pitch_mean, self.pitch_std)

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel

class TTSFinetuningDataset(TTSDataset):
    def __init__(self, 
                 dataset_path, 
                 audiopaths_and_text, 
                 text_cleaners, 
                 n_mel_channels, 
                 symbol_set='english_basic', 
                 p_arpabet=1, 
                 n_speakers=1, 
                 load_mel_from_disk=True, 
                 load_pitch_from_disk=True, 
                 pitch_mean=214.72203, 
                 pitch_std=65.72038, 
                 max_wav_value=None, 
                 sampling_rate=None, 
                 filter_length=None, 
                 hop_length=None, 
                 win_length=None, 
                 mel_fmin=None, 
                 mel_fmax=None, 
                 prepend_space_to_text=False, 
                 append_space_to_text=False, 
                 pitch_online_dir=None, 
                 betabinomial_online_dir=None, 
                 use_betabinomial_interpolator=True, 
                 pitch_online_method='pyin',
                 **ignored):
        super().__init__(dataset_path, 
                         audiopaths_and_text, 
                         text_cleaners, 
                         n_mel_channels, 
                         symbol_set=symbol_set, 
                         p_arpabet=p_arpabet, 
                         n_speakers=n_speakers, 
                         load_mel_from_disk=load_mel_from_disk, 
                         load_pitch_from_disk=load_pitch_from_disk, 
                         pitch_mean=pitch_mean, pitch_std=pitch_std, 
                         max_wav_value=max_wav_value, 
                         sampling_rate=sampling_rate, 
                         filter_length=filter_length, 
                         hop_length=hop_length, 
                         win_length=win_length, 
                         mel_fmin=mel_fmin, 
                         mel_fmax=mel_fmax, 
                         prepend_space_to_text=prepend_space_to_text, 
                         append_space_to_text=append_space_to_text, 
                         pitch_online_dir=pitch_online_dir, 
                         betabinomial_online_dir=betabinomial_online_dir, 
                         use_betabinomial_interpolator=use_betabinomial_interpolator, 
                         pitch_online_method=pitch_online_method, 
                         **ignored)

        self.speakers = []
        for audiopath, *extra, text, speaker in self.audiopaths_and_text:
            self.speakers.append(int(speaker))



class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        n_formants = batch[0][3].shape[0]
        pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype)
        energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None

        attn_prior_padded = torch.zeros(len(batch), max_target_len,
                                        max_input_len)
        attn_prior_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            prior = batch[ids_sorted_decreasing[i]][6]
            attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior

        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]

        # NLP
        lens = [len(one[-1]['subword_to_word_mapping_ids']) for one in batch]
        ancestor_lens = [len(one[-1]['ancestor_list'][i]) for one in batch for i in range(len(one[-1]['ancestor_list']))]
        next_word_relation_lens = [len(one[-1]['next_word_relation_node_idx_list'][i]) for one in batch for i in range(len(one[-1]['next_word_relation_node_idx_list']))]
        prev_word_relation_lens = [len(one[-1]['prev_word_relation_node_idx_list'][i]) for one in batch for i in range(len(one[-1]['prev_word_relation_node_idx_list']))]

        max_len = max(lens)
        ancestor_max_len = max(ancestor_lens)
        next_word_relation_max_len = max(next_word_relation_lens)
        prev_word_relation_max_len = max(prev_word_relation_lens)

        nlp_result_data = {'subword_ids': [], 'subword_segment_ids': [], 'subword_att_mask': [], 'subword_to_word_mapping_ids': [], 'forward_neighbor_list': [], 'backward_neighbor_list': [], 'forward_deprel_neighbor_list': [], 'backward_deprel_neighbor_list': [], 'phoneme_to_word_mapping_list': [], 'ancestor_list': [], 'ancestor_deprel_list': [], 'ancestor_xpos_list': [], 'ancestor_last_index_list': [],
                           'next_word_relation_node_idx_list': [], 'next_word_relation_deprel_idx_list': [], 'next_word_relation_child_parent_list': [], 'next_word_relation_upos_list': [], 'next_word_relation_xpos_list': [], 'next_word_relation_last_index_list': [], 'prev_word_relation_node_idx_list': [], 'prev_word_relation_deprel_idx_list': [], 'prev_word_relation_child_parent_list': [], 'prev_word_relation_upos_list': [], 'prev_word_relation_xpos_list': [], 'prev_word_relation_last_index_list': []}
        for cur_idx in ids_sorted_decreasing:
            one_data = batch[cur_idx][-1]
            cur_len = len(one_data['subword_ids'])
            attention_mask = []
            for i in range(cur_len):
                attention_mask.append(1)

            cur_len = len(one_data['subword_to_word_mapping_ids'])

            ancestor_last_index_list = []
            for i in range(len(one_data['ancestor_list'])):
                ancestor_last_index_list.append(len(one_data['ancestor_list'][i]) - 1)
                for j in range(len(one_data['ancestor_list'][i]), ancestor_max_len):
                    one_data['ancestor_list'][i].append(0)
                    one_data['ancestor_deprel_list'][i].append(0)
                    one_data['ancestor_xpos_list'][i].append(0)

            next_word_relation_last_index_list = []
            for i in range(len(one_data['next_word_relation_node_idx_list'])):
                next_word_relation_last_index_list.append(len(one_data['next_word_relation_node_idx_list'][i]) - 1)
                for j in range(len(one_data['next_word_relation_node_idx_list'][i]), next_word_relation_max_len):
                    one_data['next_word_relation_node_idx_list'][i].append(0)
                    one_data['next_word_relation_deprel_idx_list'][i].append(0)
                    one_data['next_word_relation_child_parent_list'][i].append(0)
                    one_data['next_word_relation_upos_list'][i].append(0)
                    one_data['next_word_relation_xpos_list'][i].append(0)

            prev_word_relation_last_index_list = []
            for i in range(len(one_data['prev_word_relation_node_idx_list'])):
                prev_word_relation_last_index_list.append(len(one_data['prev_word_relation_node_idx_list'][i]) - 1)
                for j in range(len(one_data['prev_word_relation_node_idx_list'][i]), prev_word_relation_max_len):
                    one_data['prev_word_relation_node_idx_list'][i].append(0)
                    one_data['prev_word_relation_deprel_idx_list'][i].append(0)
                    one_data['prev_word_relation_child_parent_list'][i].append(0)
                    one_data['prev_word_relation_upos_list'][i].append(0)
                    one_data['prev_word_relation_xpos_list'][i].append(0)

            for i in range(cur_len, max_len):
                one_data['subword_to_word_mapping_ids'].append([one_data['subword_to_word_mapping_ids'][-1][0] + 1])
                one_data['forward_neighbor_list'].append([one_data['forward_neighbor_list'][-1][0] + 1])
                one_data['backward_neighbor_list'].append([one_data['backward_neighbor_list'][-1][0] + 1])
                one_data['ancestor_list'].append([one_data['ancestor_list'][-1][0] + 1] + [0 for j in range(ancestor_max_len - 1)])
                one_data['ancestor_deprel_list'].append([0 for j in range(ancestor_max_len)])
                one_data['ancestor_xpos_list'].append([0 for j in range(ancestor_max_len)])
                one_data['next_word_relation_node_idx_list'].append([one_data['next_word_relation_node_idx_list'][-1][0] + 1] + [0 for j in range(next_word_relation_max_len - 1)])
                one_data['next_word_relation_deprel_idx_list'].append([0 for j in range(next_word_relation_max_len)])
                one_data['next_word_relation_child_parent_list'].append([0 for j in range(next_word_relation_max_len)])
                one_data['next_word_relation_upos_list'].append([0 for j in range(next_word_relation_max_len)])
                one_data['next_word_relation_xpos_list'].append([0 for j in range(next_word_relation_max_len)])
                one_data['prev_word_relation_node_idx_list'].append([one_data['prev_word_relation_node_idx_list'][-1][0] + 1] + [0 for j in range(prev_word_relation_max_len - 1)])
                one_data['prev_word_relation_deprel_idx_list'].append([0 for j in range(prev_word_relation_max_len)])
                one_data['prev_word_relation_child_parent_list'].append([0 for j in range(prev_word_relation_max_len)])
                one_data['prev_word_relation_upos_list'].append([0 for j in range(prev_word_relation_max_len)])
                one_data['prev_word_relation_xpos_list'].append([0 for j in range(prev_word_relation_max_len)])
                one_data['forward_deprel_neighbor_list'].append([0]) # Recursive Tag
                one_data['backward_deprel_neighbor_list'].append([0]) # Recursive Tag
                ancestor_last_index_list.append(0)
                next_word_relation_last_index_list.append(0)
                prev_word_relation_last_index_list.append(0)
                # one_data['phoneme_to_word_mapping_list'].append([1])

            for i in range(len(one_data['subword_ids']), one_data['subword_to_word_mapping_ids'][-1][0] + 1):
                one_data['subword_ids'].append(0)
                one_data['subword_segment_ids'].append(1)
                attention_mask.append(0)


            nlp_result_data['subword_ids'].append(one_data['subword_ids'])
            nlp_result_data['subword_segment_ids'].append(one_data['subword_segment_ids'])
            nlp_result_data['subword_att_mask'].append(attention_mask)

            nlp_result_data['subword_to_word_mapping_ids'].append(one_data['subword_to_word_mapping_ids'])

            nlp_result_data['forward_neighbor_list'].append(one_data['forward_neighbor_list'])
            nlp_result_data['backward_neighbor_list'].append(one_data['backward_neighbor_list'])
            nlp_result_data['forward_deprel_neighbor_list'].append(one_data['forward_deprel_neighbor_list'])
            nlp_result_data['backward_deprel_neighbor_list'].append(one_data['backward_deprel_neighbor_list'])

            nlp_result_data['phoneme_to_word_mapping_list'].append(one_data['phoneme_to_word_mapping_list'])

            nlp_result_data['ancestor_list'].append(one_data['ancestor_list'])
            nlp_result_data['ancestor_deprel_list'].append(one_data['ancestor_deprel_list'])
            nlp_result_data['ancestor_xpos_list'].append(one_data['ancestor_xpos_list'])
            nlp_result_data['ancestor_last_index_list'].append(ancestor_last_index_list)

            nlp_result_data['next_word_relation_node_idx_list'].append(one_data['next_word_relation_node_idx_list'])
            nlp_result_data['next_word_relation_deprel_idx_list'].append(one_data['next_word_relation_deprel_idx_list'])
            nlp_result_data['next_word_relation_child_parent_list'].append(one_data['next_word_relation_child_parent_list'])
            nlp_result_data['next_word_relation_upos_list'].append(one_data['next_word_relation_upos_list'])
            nlp_result_data['next_word_relation_xpos_list'].append(one_data['next_word_relation_xpos_list'])
            nlp_result_data['next_word_relation_last_index_list'].append(next_word_relation_last_index_list)

            nlp_result_data['prev_word_relation_node_idx_list'].append(one_data['prev_word_relation_node_idx_list'])
            nlp_result_data['prev_word_relation_deprel_idx_list'].append(one_data['prev_word_relation_deprel_idx_list'])
            nlp_result_data['prev_word_relation_child_parent_list'].append(one_data['prev_word_relation_child_parent_list'])
            nlp_result_data['prev_word_relation_upos_list'].append(one_data['prev_word_relation_upos_list'])
            nlp_result_data['prev_word_relation_xpos_list'].append(one_data['prev_word_relation_xpos_list'])
            nlp_result_data['prev_word_relation_last_index_list'].append(prev_word_relation_last_index_list)

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                audiopaths, nlp_result_data)


class FinetuningSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, pretraining_data_ratio=0.5, finetuning_speakers=[1], num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.speakers = dataset.speakers
        self.batch_size = batch_size
        self.pretraining_data_ratio = pretraining_data_ratio
        self.finetuning_speakers = finetuning_speakers
        self.finetuning_data_ratio = (1. - self.pretraining_data_ratio) / len(finetuning_speakers)
        
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.finetuning_speakers) + 1)]
        for i in range(len(self.speakers)):
            speaker = self.speakers[i]
            if not speaker in self.finetuning_speakers:
                buckets[0].append(i) # pre-training data
            else:
                idx_bucket = self.finetuning_speakers.index(speaker) + 1
                buckets[idx_bucket].append(i) # fine-tuning data
        
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                speaker_empty = self.finetuning_speakers.pop(i-1)
                print("[*] dataset for speaker {} is empty.".format(speaker_empty))
                
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = int(self.batch_size * self.pretraining_data_ratio) * self.num_replicas
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
                
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
                
        # calculate data counts
        data_count_for_pretraining = int(self.pretraining_data_ratio * self.batch_size)
        data_counts_for_finetuning = [sum(l) for l in np.array_split([1 for _ in range(self.batch_size - data_count_for_pretraining)], len(self.finetuning_speakers))]
        data_counts = [data_count_for_pretraining] + data_counts_for_finetuning
        
        batches = []
        max_samples = max(self.num_samples_per_bucket)
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            data_count = data_counts[i]
            len_bucket = len(bucket)
            
            # add extra samples to make it evenly divisible
            rem = max_samples - len_bucket
            augmented_indices = list(chain(*[random.sample(indices[i], len_bucket) for _ in range(rem // len_bucket)]))
            indices[i] = indices[i] + augmented_indices + indices[i][:(rem % len_bucket)]
            
            # subsample
            indices[i] = indices[i][self.rank::self.num_replicas]
            
            # batching
            for j in range(len(indices[i]) // data_counts[0]):
                batch = [bucket[idx] for idx in indices[i][j*data_count:(j+1)*data_count]]
                if i == 0:
                    batches.append(batch)
                else:
                    batches[j] += batch
            
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        return iter(self.batches)

    def __len__(self):
        return max(self.num_samples_per_bucket) // int(self.pretraining_data_ratio * self.batch_size)


def batch_to_gpu(batch):
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, audiopaths, tmp) = batch

    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()
    attn_prior = to_gpu(attn_prior).float()
    if speaker is not None:
        speaker = to_gpu(speaker).long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, audiopaths]
    y = [mel_padded, input_lengths, output_lengths]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
