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

from typing import Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import ConvReLUNorm
from common.utils import mask_from_lens
from fastpitch.alignment import b_mas, b_mas_custom, mas_width1
from fastpitch.attention import ConvAttention
from fastpitch.transformer import FFTransformer
from transformers import BertModel
import numpy as np


def regulate_len(durations, enc_out, pace: float = 1.0,
                 mel_max_len: Optional[int] = None):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0),
                               dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = ((reps_cumsum[:, :, :-1] <= range_) &
            (reps_cumsum[:, :, 1:] > range_))
    mult = mult.to(dtype)
    enc_rep = torch.matmul(mult, enc_out)

    if mel_max_len is not None:
        enc_rep = enc_rep[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return enc_rep, dec_lens


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = F.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = F.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce)
                  - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce)
                    - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems,
                            pitch_sums / pitch_nelems)
    return pitch_avg


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out


class FastPitch(nn.Module):
    def __init__(self, n_mel_channels, n_symbols, padding_idx,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size,
                 p_in_fft_dropout, p_in_fft_dropatt, p_in_fft_dropemb,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 out_fft_output_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 dur_predictor_kernel_size, dur_predictor_filter_size,
                 p_dur_predictor_dropout, dur_predictor_n_layers,
                 pitch_predictor_kernel_size, pitch_predictor_filter_size,
                 p_pitch_predictor_dropout, pitch_predictor_n_layers,
                 pitch_embedding_kernel_size,
                 energy_conditioning,
                 energy_predictor_kernel_size, energy_predictor_filter_size,
                 p_energy_predictor_dropout, energy_predictor_n_layers,
                 energy_embedding_kernel_size,
                 n_speakers, speaker_emb_weight, pitch_conditioning_formants=1):
        super(FastPitch, self).__init__()

        self.encoder = FFTransformer(
            n_layer=in_fft_n_layers, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=True,
            d_embed=symbols_embedding_dim,
            n_embed=n_symbols,
            padding_idx=padding_idx)

        if n_speakers > 1:
            self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        else:
            self.speaker_emb = None
        self.speaker_emb_weight = speaker_emb_weight

        self.duration_predictor = TemporalPredictor(
            in_fft_output_size + 768,
            filter_size=dur_predictor_filter_size,
            kernel_size=dur_predictor_kernel_size,
            dropout=p_dur_predictor_dropout, n_layers=dur_predictor_n_layers
        )

        self.decoder = FFTransformer(
            n_layer=out_fft_n_layers, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim + 768,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )

        self.pitch_predictor = TemporalPredictor(
            in_fft_output_size + 768,
            filter_size=pitch_predictor_filter_size,
            kernel_size=pitch_predictor_kernel_size,
            dropout=p_pitch_predictor_dropout, n_layers=pitch_predictor_n_layers,
            n_predictions=pitch_conditioning_formants
        )

        self.pitch_emb = nn.Conv1d(
            pitch_conditioning_formants, symbols_embedding_dim + 768,
            kernel_size=pitch_embedding_kernel_size,
            padding=int((pitch_embedding_kernel_size - 1) / 2))

        # Store values precomputed for training data within the model
        self.register_buffer('pitch_mean', torch.zeros(1))
        self.register_buffer('pitch_std', torch.zeros(1))

        self.energy_conditioning = energy_conditioning
        if energy_conditioning:
            self.energy_predictor = TemporalPredictor(
                in_fft_output_size + 768,
                filter_size=energy_predictor_filter_size,
                kernel_size=energy_predictor_kernel_size,
                dropout=p_energy_predictor_dropout,
                n_layers=energy_predictor_n_layers,
                n_predictions=1
            )

            self.energy_emb = nn.Conv1d(
                1, symbols_embedding_dim + 768,
                kernel_size=energy_embedding_kernel_size,
                padding=int((energy_embedding_kernel_size - 1) / 2))

        self.proj = nn.Linear(out_fft_output_size + 768, n_mel_channels, bias=True)

        self.attention = ConvAttention(
            n_mel_channels, 0, symbols_embedding_dim,
            use_query_proj=True, align_query_enc_type='3xconv')

        # RWEN
        self.lm = BertModel.from_pretrained('bert-base-uncased')

        self.deprel_tag_emb = nn.Embedding(num_embeddings=56, embedding_dim=128)
        self.ancestor_gru = nn.GRU(768 + 128, 768 + 128, batch_first=True)
        self.ancestor_linear = nn.Linear(768 + 128, 768)
        self.child_parent_tag_emb = nn.Embedding(num_embeddings=4, embedding_dim=128)
        self.xpos_tag_emb = nn.Embedding(num_embeddings=43, embedding_dim=128)

        self.next_word_relation_gru = nn.GRU(768 + 128 + 128, 768 + 128 + 128, batch_first=True)
        self.next_word_relation_representation_linear = nn.Linear(768 + 128 + 128, 768)

        self.prev_word_relation_gru = nn.GRU(768 + 128 + 128, 768 + 128 + 128, batch_first=True)
        self.prev_word_relation_representation_linear = nn.Linear(768 + 128 + 128, 768)

        self.awr_linear = nn.Linear(768 + 768, 768)

        self.final_linear = nn.Linear(768 + 768, 768)

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas_width1(
                    attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())
        return attn_out

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer recieve a gradient.

        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(),
                             out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.get_device())

        # with torch.no_grad():
        #     attn_out = b_mas_custom(attn, in_lens,
        #                      out_lens)
        return attn_out

    def forward(self, inputs, nlp_batch, use_gt_pitch=True, pace=1.0, max_duration=75):
        (inputs, input_lens, mel_tgt, mel_lens, pitch_dense, energy_dense,
         speaker, attn_prior, audiopaths) = inputs

        mel_max_len = mel_tgt.size(2)

        # Calculate speaker embedding
        if self.speaker_emb is None:
            spk_emb = 0
        else:
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Word-level Average Pooling
        pooled_tensor = self.get_avg_pooled_tensor(nlp_batch['subword_ids'], nlp_batch['subword_segment_ids'], nlp_batch['subword_att_mask'], nlp_batch['subword_to_word_mapping_ids'])

        # SRE method
        ancestor_out = self.ancestor_net(pooled_tensor, nlp_batch['ancestor_list'], nlp_batch['ancestor_deprel_list'], nlp_batch['ancestor_xpos_list'], nlp_batch['ancestor_last_index_list'], self.deprel_tag_emb, self.xpos_tag_emb, self.ancestor_gru, self.ancestor_linear)

        # AWRE method
        nwr_representation = self.adjacent_word_relation_net(pooled_tensor, nlp_batch['next_word_relation_node_idx_list'], nlp_batch['next_word_relation_deprel_idx_list'], nlp_batch['next_word_relation_child_parent_list'], nlp_batch['next_word_relation_xpos_list'], nlp_batch['next_word_relation_last_index_list'],
                                                                         self.deprel_tag_emb, self.child_parent_tag_emb, self.xpos_tag_emb, self.next_word_relation_gru, self.next_word_relation_representation_linear)
        pwr_representation = self.adjacent_word_relation_net(pooled_tensor, nlp_batch['prev_word_relation_node_idx_list'], nlp_batch['prev_word_relation_deprel_idx_list'], nlp_batch['prev_word_relation_child_parent_list'], nlp_batch['prev_word_relation_xpos_list'], nlp_batch['prev_word_relation_last_index_list'],
                                                                         self.deprel_tag_emb, self.child_parent_tag_emb, self.xpos_tag_emb, self.prev_word_relation_gru, self.prev_word_relation_representation_linear)
        awr_representation = self.awr_linear(torch.cat((nwr_representation, pwr_representation), -1))

        word_level_semantic_representation = self.final_linear(torch.cat((ancestor_out, awr_representation), -1))


        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Upsampling
        phoneme_to_word_semantic_representation = self.repeat_phoneme_to_word(enc_out, word_level_semantic_representation, nlp_batch['phoneme_to_word_mapping_list'])
        enc_out = torch.cat((enc_out, phoneme_to_word_semantic_representation), -1)

        # Alignment
        text_emb = self.encoder.word_emb(inputs)

        # make sure to do the alignments before folding
        attn_mask = mask_from_lens(input_lens)[..., None] == 0
        # attn_mask should be 1 for unused timesteps in the text_enc_w_spkvec tensor

        attn_soft, attn_logprob = self.attention(
            mel_tgt, text_emb.permute(0, 2, 1), mel_lens, attn_mask,
            key_lens=input_lens, keys_encoded=enc_out, attn_prior=attn_prior)

        attn_hard = self.binarize_attention_parallel(
            attn_soft, input_lens, mel_lens)


        # Viterbi --> durations
        attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        dur_tgt = attn_hard_dur

        assert torch.all(torch.eq(dur_tgt.sum(dim=1), mel_lens)), f"{torch.eq(dur_tgt.sum(dim=1), mel_lens)} / {audiopaths}"

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Predict pitch
        pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

        # Average pitch over characters
        pitch_tgt = average_pitch(pitch_dense, dur_tgt)

        if use_gt_pitch and pitch_tgt is not None:
            pitch_emb = self.pitch_emb(pitch_tgt)
        else:
            pitch_emb = self.pitch_emb(pitch_pred)
        enc_out = enc_out + pitch_emb.transpose(1, 2)

        # Predict energy
        if self.energy_conditioning:
            energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)

            # Average energy over characters
            energy_tgt = average_pitch(energy_dense.unsqueeze(1), dur_tgt)
            energy_tgt = torch.log(1.0 + energy_tgt)

            energy_emb = self.energy_emb(energy_tgt)
            energy_tgt = energy_tgt.squeeze(1)
            enc_out = enc_out + energy_emb.transpose(1, 2)
        else:
            energy_pred = None
            energy_tgt = None

        len_regulated, dec_lens = regulate_len(
            dur_tgt, enc_out, pace, mel_max_len)

        # Output FFT
        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)

        return (mel_out, dec_mask, dur_pred, log_dur_pred, pitch_pred,
                pitch_tgt, energy_pred, energy_tgt, attn_soft, attn_hard,
                attn_hard_dur, attn_logprob)

    def infer(self, inputs, nlp_batch, pace=1.0, dur_tgt=None, pitch_tgt=None,
              energy_tgt=None, pitch_transform=None, max_duration=75,
              speaker=0):

        if self.speaker_emb is None:
            spk_emb = 0
        else:
            speaker = (torch.ones(inputs.size(0)).long().to(inputs.device)
                       * speaker)
            spk_emb = self.speaker_emb(speaker).unsqueeze(1)
            spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        enc_out, enc_mask = self.encoder(inputs, conditioning=spk_emb)

        # Word-level Average Pooling
        pooled_tensor = self.get_avg_pooled_tensor(nlp_batch['subword_ids'], nlp_batch['subword_segment_ids'], nlp_batch['subword_att_mask'], nlp_batch['subword_to_word_mapping_ids'])

        # SRE
        ancestor_out = self.ancestor_net(pooled_tensor, nlp_batch['ancestor_list'], nlp_batch['ancestor_deprel_list'], nlp_batch['ancestor_xpos_list'], nlp_batch['ancestor_last_index_list'], self.deprel_tag_emb, self.xpos_tag_emb, self.ancestor_gru, self.ancestor_linear)

        # AWRE
        nwr_representation = self.adjacent_word_relation_net(pooled_tensor, nlp_batch['next_word_relation_node_idx_list'], nlp_batch['next_word_relation_deprel_idx_list'], nlp_batch['next_word_relation_child_parent_list'], nlp_batch['next_word_relation_xpos_list'], nlp_batch['next_word_relation_last_index_list'],
                                                                         self.deprel_tag_emb, self.child_parent_tag_emb, self.xpos_tag_emb, self.next_word_relation_gru, self.next_word_relation_representation_linear)
        pwr_representation = self.adjacent_word_relation_net(pooled_tensor, nlp_batch['prev_word_relation_node_idx_list'], nlp_batch['prev_word_relation_deprel_idx_list'], nlp_batch['prev_word_relation_child_parent_list'], nlp_batch['prev_word_relation_xpos_list'], nlp_batch['prev_word_relation_last_index_list'],
                                                                         self.deprel_tag_emb, self.child_parent_tag_emb, self.xpos_tag_emb, self.prev_word_relation_gru, self.prev_word_relation_representation_linear)
        awr_representation = self.awr_linear(torch.cat((nwr_representation, pwr_representation), -1))

        word_level_semantic_representation = self.final_linear(torch.cat((ancestor_out, awr_representation), -1))

        # Upsampling
        phoneme_to_word_semantic_representation = self.repeat_phoneme_to_word(enc_out, word_level_semantic_representation, nlp_batch['phoneme_to_word_mapping_list'])
        enc_out = torch.cat((enc_out, phoneme_to_word_semantic_representation), -1)

        # Predict durations
        log_dur_pred = self.duration_predictor(enc_out, enc_mask).squeeze(-1)
        dur_pred = torch.clamp(torch.exp(log_dur_pred) - 1, 0, max_duration)

        # Pitch over chars
        pitch_pred = self.pitch_predictor(enc_out, enc_mask).permute(0, 2, 1)

        if pitch_transform is not None:
            if self.pitch_std[0] == 0.0:
                # XXX LJSpeech-1.1 defaults
                mean, std = 218.14, 67.24
            else:
                mean, std = self.pitch_mean[0], self.pitch_std[0]
            pitch_pred = pitch_transform(pitch_pred, enc_mask.sum(dim=(1,2)),
                                         mean, std)
        if pitch_tgt is None:
            pitch_emb = self.pitch_emb(pitch_pred).transpose(1, 2)
        else:
            pitch_emb = self.pitch_emb(pitch_tgt).transpose(1, 2)

        enc_out = enc_out + pitch_emb

        # Predict energy
        if self.energy_conditioning:

            if energy_tgt is None:
                energy_pred = self.energy_predictor(enc_out, enc_mask).squeeze(-1)
                energy_emb = self.energy_emb(energy_pred.unsqueeze(1)).transpose(1, 2)
            else:
                energy_emb = self.energy_emb(energy_tgt).transpose(1, 2)

            enc_out = enc_out + energy_emb
        else:
            energy_pred = None

        len_regulated, dec_lens = regulate_len(
            dur_pred if dur_tgt is None else dur_tgt,
            enc_out, pace, mel_max_len=None)

        dec_out, dec_mask = self.decoder(len_regulated, dec_lens)
        mel_out = self.proj(dec_out)
        # mel_lens = dec_mask.squeeze(2).sum(axis=1).long()
        mel_out = mel_out.permute(0, 2, 1)  # For inference.py
        return mel_out, dec_lens, dur_pred, pitch_pred, energy_pred

    # Word-level Average Pooling
    def get_avg_pooled_tensor(self, enc_input_ids, attention_mask, token_type_ids, mapping_ids):
        batch_tensor = None
        for i in range(len(mapping_ids)):
            each_input_ids = torch.tensor(enc_input_ids[i]).long().to('cuda').unsqueeze(0)
            each_attention_mask = torch.tensor(attention_mask[i]).long().to('cuda').unsqueeze(0)
            each_token_type_ids = torch.tensor(token_type_ids[i]).long().to('cuda').unsqueeze(0)
            lm_output = self.lm(input_ids=each_input_ids, attention_mask=each_attention_mask, token_type_ids=each_token_type_ids)[0]

            # average pooling
            result_tensor = None
            for j in range(len(mapping_ids[i])):
                if len(mapping_ids[i][j]) == 1:
                    if result_tensor is None:
                        result_tensor = lm_output[:, mapping_ids[i][j][0], :].unsqueeze(0)
                    else:
                        result_tensor = torch.cat((result_tensor, lm_output[:, mapping_ids[i][j][0], :].unsqueeze(0)), 1)
                else:
                    tmp_tensor = lm_output[:, mapping_ids[i][j][0]:mapping_ids[i][j][-1] + 1, :]
                    average_pool = torch.mean(tmp_tensor, 1, True)
                    if result_tensor is None:
                        result_tensor = average_pool
                    else:
                        result_tensor = torch.cat((result_tensor, average_pool), 1)

            if batch_tensor is None:
                batch_tensor = result_tensor
            else:
                batch_tensor = torch.cat((batch_tensor, result_tensor), 0)
        return batch_tensor

    # Upsampling
    def repeat_phoneme_to_word(self, enc_out, word_level_semantic_representation, phoneme_to_word_mapping_list):
        '''
        cls -> preprend space
        word -> phoneme word + space
        sep -> x
        '''
        batch_result_tensor = None
        max_len = enc_out.shape[1]
        for b in range(word_level_semantic_representation.shape[0]):
            one_result_tensor = word_level_semantic_representation[b, 0, :].unsqueeze(0).unsqueeze(0)
            for i in range(len(phoneme_to_word_mapping_list[b])):
                cur_tensor = word_level_semantic_representation[b, i+1, :].unsqueeze(0).unsqueeze(0).repeat((1, phoneme_to_word_mapping_list[b][i] + 1, 1))
                one_result_tensor = torch.cat((one_result_tensor, cur_tensor), 1)

            # add padding
            assert one_result_tensor.shape[1] <= max_len
            one_result_tensor = F.pad(one_result_tensor, (0, 0, 0, max_len - one_result_tensor.shape[1], 0, 0), mode='constant', value=0)

            if batch_result_tensor == None:
                batch_result_tensor = one_result_tensor
            else:
                batch_result_tensor = torch.cat((batch_result_tensor, one_result_tensor), 0)
        return batch_result_tensor

    # SRE method
    def ancestor_net(self, pooled_tensor, ancestor_list, ancestor_deprel_list, ancestor_pos_list, ancestor_last_index_list, ancestor_tag_emb, pos_tag_emb, ancestor_gru, ancestor_linear):
        cur_device = pooled_tensor.device
        gru_out_batch = None
        for i in range(pooled_tensor.shape[1]):
            cur_seq_gru_input_batch = None
            cur_tag_emb_batch = None
            for b in range(pooled_tensor.shape[0]):
                cur_batch_lm_output = pooled_tensor[b, :, :]
                indices = torch.tensor(ancestor_list[b][i]).to(cur_device)
                cur_tensor = torch.index_select(cur_batch_lm_output, 0, indices).unsqueeze(0)

                cur_deprel_tag_emb = ancestor_tag_emb(torch.tensor(ancestor_deprel_list[b][i]).to(cur_device)).unsqueeze(0)
                cur_tag_emb = cur_deprel_tag_emb

                if b == 0:
                    cur_seq_gru_input_batch = cur_tensor
                    cur_tag_emb_batch = cur_tag_emb
                else:
                    cur_seq_gru_input_batch = torch.cat((cur_seq_gru_input_batch, cur_tensor), 0)
                    cur_tag_emb_batch = torch.cat((cur_tag_emb_batch, cur_tag_emb), 0)
            gru_in = torch.cat((cur_seq_gru_input_batch, cur_tag_emb_batch), -1)
            gru_out, _ = ancestor_gru(gru_in)

            # get gru last hidden
            cur_gru_out_batch = None
            for b in range(pooled_tensor.shape[0]):
                indices = torch.tensor([ancestor_last_index_list[b][i]]).to(cur_device)
                cur_gru_out = torch.index_select(gru_out[b, :, :], 0, indices).unsqueeze(0)
                if b == 0:
                    cur_gru_out_batch = cur_gru_out
                else:
                    cur_gru_out_batch = torch.cat((cur_gru_out_batch, cur_gru_out), 0)

            if i == 0:
                gru_out_batch = cur_gru_out_batch
            else:
                gru_out_batch = torch.cat((gru_out_batch, cur_gru_out_batch), 1)

        output = ancestor_linear(gru_out_batch)
        return output

    # AWRE method
    def adjacent_word_relation_net(self, pooled_tensor, node_idx_list, deprel_list, child_parent_list, pos_list, last_index_list, deprel_tag_emb, child_parent_tag_emb, pos_tag_emb, gru, representation_layer):
        # emb + pooled tensor -> gru last hidden 둘 다 concat -> linear 태워서 차원 축소
        # 데이터 잘 들어갔나 확인
        cur_device = pooled_tensor.device
        gru_out_batch = None
        for i in range(pooled_tensor.shape[1]):
            cur_seq_gru_input_batch = None
            cur_tag_emb_batch = None
            for b in range(pooled_tensor.shape[0]):
                cur_batch_lm_output = pooled_tensor[b, :, :]
                indices = torch.tensor(node_idx_list[b][i]).to(cur_device)
                cur_tensor = torch.index_select(cur_batch_lm_output, 0, indices).unsqueeze(0)

                cur_deprel_tag_emb = deprel_tag_emb(torch.tensor(deprel_list[b][i]).to(cur_device)).unsqueeze(0)
                cur_parent_tag_emb = child_parent_tag_emb(torch.tensor(child_parent_list[b][i]).to(cur_device)).unsqueeze(0)

                cur_tag_emb = torch.cat((cur_deprel_tag_emb, cur_parent_tag_emb), -1)

                if b == 0:
                    cur_seq_gru_input_batch = cur_tensor
                    cur_tag_emb_batch = cur_tag_emb
                else:
                    cur_seq_gru_input_batch = torch.cat((cur_seq_gru_input_batch, cur_tensor), 0)
                    cur_tag_emb_batch = torch.cat((cur_tag_emb_batch, cur_tag_emb), 0)
            gru_in = torch.cat((cur_seq_gru_input_batch, cur_tag_emb_batch), -1)
            gru_out, _ = gru(gru_in)

            # get gru last hidden
            cur_gru_out_batch = None
            for b in range(pooled_tensor.shape[0]):
                indices = torch.tensor([last_index_list[b][i]]).to(cur_device)
                cur_gru_out = torch.index_select(gru_out[b, :, :], 0, indices).unsqueeze(0)
                if b == 0:
                    cur_gru_out_batch = cur_gru_out
                else:
                    cur_gru_out_batch = torch.cat((cur_gru_out_batch, cur_gru_out), 0)

            if i == 0:
                gru_out_batch = cur_gru_out_batch
            else:
                gru_out_batch = torch.cat((gru_out_batch, cur_gru_out_batch), 1)

        output = representation_layer(gru_out_batch)
        return output