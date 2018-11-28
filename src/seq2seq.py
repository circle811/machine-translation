"""
note:
    if rnn_unit is 'rnn' or 'gru' then
        hidden(layer_number * direction_number, batch_size, hidden_size)
        means
        (layer_number * direction_number, batch_size, hidden_size)
    if rnn_unit is 'lstm' then
        hidden(layer_number * direction_number, batch_size, hidden_size)
        means
        tuple
            (layer_number * direction_number, batch_size, hidden_size)
            (layer_number * direction_number, batch_size, hidden_size)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

__all__ = ['Seq2seq']

INF = float('inf')

PAD, SOS, EOS = range(3)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_length(full_words):
    """
    :param full_words: (batch_size, max_length)
    :return:
            (batch_size,)
    """

    _, max_length = full_words.shape
    have_eos, i_eos = torch.max(full_words == EOS, dim=1)
    length = torch.where(have_eos, i_eos + 1, torch.tensor(max_length, device=device))
    return length


def random_sample(p):
    """
    random sample form range(number) with probability p
    :param p: (length, number)
    :return:
            (length,)
    """

    length, _ = p.shape
    s = torch.zeros(length, dtype=torch.int64, device=device)
    for i in range(length):
        s[i] = torch.multinomial(p[i], 1)
    return s


class Encoder(nn.Module):
    def __init__(self, rnn_type, layer_number, source_word_number, source_embedding_size, source_hidden_size):
        """
        :param rnn_type:
        :param layer_number:
        :param source_word_number:
        :param source_embedding_size:
        :param source_hidden_size:
        """

        super().__init__()
        self.embedding = nn.Embedding(source_word_number, source_embedding_size, padding_idx=PAD)
        self.rnn = rnn_type(source_embedding_size, source_hidden_size, layer_number, bidirectional=True)

    def forward(self, source_full_words, source_length):
        """
        :param source_full_words: (batch_size, source_max_length)
        :param source_length:     (batch_size,)
        :return:
                (batch_size, source_max_length, 2 * source_hidden_size)
                hidden(layer_number * 2, batch_size, source_hidden_size)
        """

        e = self.embedding(source_full_words)
        o, source_hidden = self.rnn(pack_padded_sequence(e, source_length, batch_first=True))
        source_full_output, _ = pad_packed_sequence(o, batch_first=True)
        return source_full_output, source_hidden


class Converter(nn.Module):
    def __init__(self, is_lstm, layer_number, source_hidden_size, target_hidden_size):
        """
        :param is_lstm:
        :param layer_number:
        :param source_hidden_size:
        :param target_hidden_size:
        """

        super().__init__()
        self.is_lstm = is_lstm
        self.weight = nn.Parameter(torch.Tensor(layer_number, 2 * source_hidden_size, target_hidden_size))
        self.bias = nn.Parameter(torch.Tensor(layer_number, 1, target_hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, source_hidden):
        """
        :param source_hidden: hidden(layer_number * 2, batch_size, source_hidden_size)
        :return:
                hidden(layer_number, batch_size, target_hidden_size)
        """

        if self.is_lstm:
            sh, _ = source_hidden
            th = self._convert(sh)
            tc = torch.zeros_like(th)
            target_hidden = (th, tc)
        else:
            target_hidden = self._convert(source_hidden)
        return target_hidden

    def _convert(self, sh):
        """
        :param sh: (layer_number * 2, batch_size, source_hidden_size)
        :return:
                (layer_number, batch_size, target_hidden_size)
        """

        layer_number_2, batch_size, source_hidden_size = sh.shape
        sh_transposed = (sh.view(layer_number_2 // 2, 2, batch_size, source_hidden_size)
                         .transpose(1, 2)
                         .contiguous()
                         .view(layer_number_2 // 2, batch_size, 2 * source_hidden_size))
        th = torch.tanh(sh_transposed @ self.weight + self.bias)
        return th

    def extra_repr(self):
        layer_number, source_hidden_size_2, target_hidden_size = self.weight.shape
        return 'is_lstm={}, layer_number={}, source_hidden_size={}, target_hidden_size={}'.format(
            self.is_lstm, layer_number, source_hidden_size_2 // 2, target_hidden_size)


class AttentionBase(nn.Module):
    def forward(self, source_length, source_full_output, target_output):
        """
        :param source_length:      (batch_size,)
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_output:      (batch_size, target_hidden_size)
        :return:
                (batch_size, source_max_length)
        """

        _, source_max_length, _ = source_full_output.shape
        s = self._calculate_score(source_full_output, target_output)
        s_masked = torch.where(torch.arange(source_max_length, device=device) < source_length[:, None],
                               s, torch.tensor(-INF, device=device))
        a = F.softmax(s_masked, dim=1)
        return a

    def _calculate_score(self, source_full_output, target_output):
        """
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_output:      (batch_size, target_hidden_size)
        :return:
                (batch_size, source_max_length)
        """

        raise NotImplemented


class AttentionDot(AttentionBase):
    def __init__(self, context_size, target_hidden_size):
        """
        :param context_size:
        :param target_hidden_size:
        """

        super().__init__()
        assert context_size == target_hidden_size

    def _calculate_score(self, source_full_output, target_output):
        """
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_output:      (batch_size, target_hidden_size)
        :return:
                (batch_size, source_max_length)
        """

        s = (source_full_output @ target_output[..., None]).squeeze(dim=2)
        return s


class AttentionBilinear(AttentionBase):
    def __init__(self, context_size, target_hidden_size):
        """
        :param context_size:
        :param target_hidden_size:
        """

        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(target_hidden_size, context_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[0])
        self.weight.data.uniform_(-stdv, stdv)

    def _calculate_score(self, source_full_output, target_output):
        """
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_output:      (batch_size, target_hidden_size)
        :return:
                (batch_size, source_max_length)
        """

        s = (source_full_output @ (target_output @ self.weight)[..., None]).squeeze(dim=2)
        return s

    def extra_repr(self):
        target_hidden_size, context_size = self.weight.shape
        return 'context_size={}, target_hidden_size={}'.format(context_size, target_hidden_size)


class AttentionMLP(AttentionBase):
    def __init__(self, context_size, target_hidden_size, attention_hidden_size):
        """
        :param context_size:
        :param target_hidden_size:
        :param attention_hidden_size:
        """

        super().__init__()
        self.linear0 = nn.Linear(context_size + target_hidden_size, attention_hidden_size)
        self.linear1 = nn.Linear(attention_hidden_size, 1)

    def _calculate_score(self, source_full_output, target_output):
        """
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_output:      (batch_size, target_hidden_size)
        :return:
                (batch_size, source_max_length)
        """

        _, source_max_length, _ = source_full_output.shape
        batch_size, _ = target_output.shape
        so = source_full_output.expand(batch_size, -1, -1)
        to = target_output[:, None].expand(-1, source_max_length, -1)
        so_to = torch.cat([so, to], dim=2)
        s = self.linear1(torch.tanh(self.linear0(so_to))).squeeze(dim=2)
        return s


class Node:
    __slots__ = ['previous', 'log_p', 'target_hidden', 'context', 'target_words', 'attention']

    def __init__(self, previous, log_p, target_hidden, context, target_words, attention):
        """
        :param previous:      Node
        :param log_p:         ()
        :param target_hidden: hidden(layer_number, target_hidden_size)
        :param context:       (context_size,)
        :param target_words:  ()
        :param attention:     (source_max_length,)
        """

        self.previous = previous
        self.log_p = log_p
        self.target_hidden = target_hidden
        self.context = context
        self.target_words = target_words
        self.attention = attention

    def to_list(self):
        """
        :return:
                list of Node
        """

        r = []
        p = self
        while p.previous is not None:
            r.append(p)
            p = p.previous
        r.reverse()
        return r


class Decoder(nn.Module):
    def __init__(self, is_lstm, rnn_type, attention_object, layer_number, context_size,
                 target_word_number, target_embedding_size, target_hidden_size):
        """
        :param is_lstm:
        :param rnn_type:
        :param attention_object:
        :param layer_number:
        :param context_size:
        :param target_word_number:
        :param target_embedding_size:
        :param target_hidden_size:
        """

        super().__init__()
        self.is_lstm = is_lstm
        self.attention_object = attention_object
        self.embedding = nn.Embedding(target_word_number, target_embedding_size, padding_idx=PAD)
        self.rnn = rnn_type(target_embedding_size + context_size, target_hidden_size, layer_number)
        self.linear = nn.Linear(target_hidden_size + context_size, target_word_number)

    def forward(self, source_length, source_full_output, target_hidden, context, target_words):
        """
        :param source_length:      (batch_size,)
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_hidden:      hidden(layer_number, batch_size, target_hidden_size)
        :param context:            (batch_size, context_size)
        :param target_words:       (batch_size,)
        :return:
                hidden(layer_number, batch_size, target_hidden_size)
                (batch_size, context_size)
                (batch_size, target_word_number)
                (batch_size, source_max_length)
        """

        e = self.embedding(target_words)
        e_c = torch.cat([e, context], dim=1)
        o, next_target_hidden = self.rnn(e_c[None], target_hidden)
        target_output = o.squeeze(dim=0)
        attention = self.attention_object(source_length, source_full_output, target_output)
        next_context = (attention[:, None] @ source_full_output).squeeze(dim=1)
        to_nc = torch.cat([target_output, next_context], dim=1)
        next_target_words_score = self.linear(to_nc)
        return next_target_hidden, next_context, next_target_words_score, attention.detach()

    def forward_multi(self, source_length, source_full_output, target_hidden, target_max_length, get_next_target_words,
                      need_attention=False):
        """
        :param source_length:         (batch_size,)
        :param source_full_output:    (batch_size, source_max_length, context_size)
        :param target_hidden:         hidden(layer_number, batch_size, target_hidden_size)
        :param target_max_length:     int
        :param get_next_target_words: (int, (batch_size, target_word_number)) -> (batch_size,)
        :param need_attention:        bool
        :return:
                (batch_size, target_max_length)
                (batch_size, target_max_length, target_word_number)
                (batch_size, target_max_length, source_max_length) or None
        """

        batch_size, _, context_size = source_full_output.shape
        target_words_list = []
        target_words_score_list = []
        attention_list = []
        context = torch.zeros(batch_size, context_size, device=device)
        target_words = torch.full((batch_size,), SOS, dtype=torch.int64, device=device)

        for i in range(target_max_length):
            next_target_hidden, next_context, next_target_words_score, attention = self(
                source_length, source_full_output, target_hidden, context, target_words)
            next_target_words = get_next_target_words(i, next_target_words_score)
            target_words_list.append(next_target_words)
            target_words_score_list.append(next_target_words_score)
            attention_list.append(attention)
            target_hidden, context, target_words = next_target_hidden, next_context, next_target_words

        target_full_words_o = torch.stack(target_words_list, dim=1)
        target_full_words_score = torch.stack(target_words_score_list, dim=1)
        if need_attention:
            full_attention = torch.stack(attention_list, dim=1)
        else:
            full_attention = None

        return target_full_words_o, target_full_words_score, full_attention

    def cal_loss(self, source_length, source_full_output, target_hidden, target_full_words, target_length,
                 teacher_force=True):
        """
        :param source_length:      (batch_size,)
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_hidden:      hidden(layer_number, batch_size, target_hidden_size)
        :param target_full_words:  (batch_size, target_max_length)
        :param target_length:      (batch_size,)
        :param teacher_force:      bool
        :return:
                ()
        """

        _, target_max_length = target_full_words.shape

        if teacher_force:
            _, target_full_words_score, _ = self.forward_multi(
                source_length, source_full_output, target_hidden, target_max_length,
                lambda i, next_target_words_score: target_full_words[:, i])
            target_length_o = target_length
        else:
            target_full_words_o, target_full_words_score, _ = self.forward_multi(
                source_length, source_full_output, target_hidden, target_max_length,
                lambda i, next_target_words_score: torch.argmax(next_target_words_score, dim=1))
            target_length_o = torch.min(target_length, get_length(target_full_words_o))

        mask = torch.arange(target_max_length, device=device) < target_length_o[:, None]
        loss = F.cross_entropy(target_full_words_score[mask], target_full_words[mask], reduction='sum')

        return loss

    def random_sample(self, source_length, source_full_output, target_hidden, target_max_length, need_attention=False):
        """
        :param source_length:      (batch_size,)
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_hidden:      hidden(layer_number, batch_size, target_hidden_size)
        :param target_max_length:  int
        :param need_attention:     bool
        :return:
                (batch_size, target_max_length)
                (batch_size, target_max_length, target_word_number)
                (batch_size, target_max_length, source_max_length) or None
        """

        return self.forward_multi(
            source_length, source_full_output, target_hidden, target_max_length,
            lambda i, next_target_words_score: random_sample(F.softmax(next_target_words_score, dim=1)),
            need_attention=need_attention)

    def greedy_search(self, source_length, source_full_output, target_hidden, target_max_length, need_attention=False):
        """
        :param source_length:      (batch_size,)
        :param source_full_output: (batch_size, source_max_length, context_size)
        :param target_hidden:      hidden(layer_number, batch_size, target_hidden_size)
        :param target_max_length:  int
        :param need_attention:     bool
        :return:
                (batch_size, target_max_length)
                (batch_size, target_max_length, target_word_number)
                (batch_size, target_max_length, source_max_length) or None
        """

        return self.forward_multi(
            source_length, source_full_output, target_hidden, target_max_length,
            lambda i, next_target_words_score: torch.argmax(next_target_words_score, dim=1),
            need_attention=need_attention)

    def _beam_expand(self, source_length, source_full_output, nodes, beam_size):
        """
        :param source_length:      (1,)
        :param source_full_output: (1, source_max_length, context_size)
        :param nodes:              list of Node
        :param beam_size:          int
        :return:
                list of Node
                list of Node
        """

        if self.is_lstm:
            target_hidden = (torch.stack([n.target_hidden[0] for n in nodes], dim=1),
                             torch.stack([n.target_hidden[1] for n in nodes], dim=1))
        else:
            target_hidden = torch.stack([n.target_hidden for n in nodes], dim=1)
        context = torch.stack([n.context for n in nodes], dim=0)
        target_words = torch.stack([n.target_words for n in nodes], dim=0)

        next_target_hidden, next_context, next_target_words_score, attention = self(
            source_length, source_full_output, target_hidden, context, target_words)
        _, target_word_number = next_target_words_score.shape
        log_p = torch.stack([n.log_p for n in nodes], dim=0)
        next_log_p = log_p[:, None] + F.log_softmax(next_target_words_score, dim=1)

        next_nodes = []
        final_nodes = []
        for v, i in zip(*torch.topk(torch.flatten(next_log_p), beam_size)):
            j = i // target_word_number
            k = i % target_word_number
            n = Node(previous=nodes[j],
                     log_p=v,
                     target_hidden=((next_target_hidden[0][:, j], next_target_hidden[1][:, j])
                                    if self.is_lstm else
                                    next_target_hidden[:, j]),
                     context=next_context[j],
                     target_words=k,
                     attention=attention[j])
            if k != EOS:
                next_nodes.append(n)
            else:
                final_nodes.append(n)

        return next_nodes, final_nodes

    def beam_search(self, source_length, source_full_output, target_hidden, target_max_length, beam_size,
                    need_attention=False):
        """
        :param source_length:      (1,)
        :param source_full_output: (1, source_max_length, context_size)
        :param target_hidden:      hidden(layer_number, 1, target_hidden_size)
        :param target_max_length:  int
        :param beam_size:          int
        :param need_attention:     bool
        :return:
                (beam_size, *)
                (beam_size, *, source_max_length) or None
        """

        _, _, context_size = source_full_output.shape
        sos = Node(previous=None,
                   log_p=torch.tensor(0.0, device=device),
                   target_hidden=((target_hidden[0][:, 0], target_hidden[1][:, 0])
                                  if self.is_lstm else
                                  target_hidden[:, 0]),
                   context=torch.zeros(context_size, device=device),
                   target_words=torch.tensor(SOS, device=device),
                   attention=None)

        nodes = [sos]
        result_nodes = []
        for i in range(target_max_length):
            nodes, final_nodes = self._beam_expand(source_length, source_full_output, nodes, beam_size)
            result_nodes.extend(final_nodes)
            if len(nodes) == 0:
                break
        result_nodes.extend(nodes)
        result_nodes = sorted(result_nodes, key=lambda n: n.log_p, reverse=True)[:beam_size]

        target_sentences_o = []
        full_attention = [] if need_attention else None
        for n in result_nodes[:beam_size]:
            n_list = n.to_list()
            target_sentences_o.append(torch.stack([m.target_words for m in n_list], dim=0).cpu())
            if need_attention:
                full_attention.append(torch.stack([m.attention for m in n_list], dim=0).cpu())

        return target_sentences_o, full_attention


class Seq2seq(nn.Module):
    rnn_type_dict = {
        'rnn': nn.RNN,
        'gru': nn.GRU,
        'lstm': nn.LSTM
    }

    attention_type_dict = {
        'dot': AttentionDot,
        'bilinear': AttentionBilinear,
        'mlp': AttentionMLP
    }

    def __init__(self, rnn_unit, attention_unit, layer_number,
                 source_word_number, source_embedding_size, source_hidden_size,
                 target_word_number, target_embedding_size, target_hidden_size,
                 attention_hidden_size=None):
        """
        :param rnn_unit:              'rnn', 'gru' or 'lstm'
        :param attention_unit:        'dot', 'bilinear' or 'mlp'
        :param layer_number:
        :param source_word_number:
        :param source_embedding_size:
        :param source_hidden_size:
        :param target_word_number:
        :param target_embedding_size:
        :param target_hidden_size:
        :param attention_hidden_size:
        """

        super().__init__()
        is_lstm = rnn_unit == 'lstm'
        rnn_type = Seq2seq.rnn_type_dict[rnn_unit]
        attention_type = Seq2seq.attention_type_dict[attention_unit]
        context_size = 2 * source_hidden_size
        if attention_unit == 'mlp':
            attention_object = attention_type(context_size, target_hidden_size, attention_hidden_size)
        else:
            attention_object = attention_type(context_size, target_hidden_size)
        self.encoder = Encoder(rnn_type, layer_number, source_word_number, source_embedding_size, source_hidden_size)
        self.converter = Converter(is_lstm, layer_number, source_hidden_size, target_hidden_size)
        self.decoder = Decoder(is_lstm, rnn_type, attention_object, layer_number, context_size,
                               target_word_number, target_embedding_size, target_hidden_size)
        self.to(device)

    def forward(self, source_sentences):
        """
        :param source_sentences: (batch_size, *)
        :return:
                (batch_size,)
                (batch_size,)
                (batch_size, source_max_length, context_size)
                hidden(layer_number, batch_size, target_hidden_size)
        """

        source_length, index = torch.sort(torch.tensor([len(s) for s in source_sentences]), descending=True)
        source_length = source_length.to(device)
        source_sentences_sorted = [source_sentences[i] for i in index]
        source_full_words = pad_sequence(source_sentences_sorted, batch_first=True).to(device)
        source_full_output, source_hidden = self.encoder(source_full_words, source_length)
        target_hidden = self.converter(source_hidden)
        return index, source_length, source_full_output, target_hidden

    def cal_loss(self, source_sentences, target_sentences, teacher_force=True):
        """
        :param source_sentences: (batch_size, *)
        :param target_sentences: (batch_size, *)
        :param teacher_force:    bool
        :return:
                ()
        """

        index, source_length, source_full_output, target_hidden = self(source_sentences)
        target_sentences_sorted = [target_sentences[i] for i in index]
        target_full_words = pad_sequence(target_sentences_sorted, batch_first=True).to(device)
        target_length = torch.tensor([len(s) for s in target_sentences_sorted], device=device)
        loss = self.decoder.cal_loss(source_length, source_full_output, target_hidden, target_full_words, target_length,
                                     teacher_force=teacher_force)
        return loss

    def greedy_search(self, source_sentences, target_max_length, need_attention=False):
        """
        :param source_sentences:  (batch_size, *)
        :param target_max_length: int
        :param need_attention:    bool
        :return:
                (batch_size, *)
                (batch_size, target_max_length, source_max_length) or None
        """

        index, source_length, source_full_output, target_hidden = self(source_sentences)
        target_full_words_o_sorted, _, full_attention_sorted = self.decoder.greedy_search(
            source_length, source_full_output, target_hidden, target_max_length, need_attention=need_attention)

        inverse_index = torch.zeros_like(index)
        inverse_index[index] = torch.arange(index.shape[0])

        target_sentences_o = list(target_full_words_o_sorted.cpu()[inverse_index])
        if need_attention:
            full_attention = list(full_attention_sorted.cpu()[inverse_index])
        else:
            full_attention = None

        return target_sentences_o, full_attention

    def beam_search(self, source_sentences, target_max_length, beam_size, need_attention=False):
        """
        :param source_sentences:  (1, *)
        :param target_max_length: int
        :param beam_size:         int
        :param need_attention:    bool
        :return:
                (beam_size, *)
                (beam_size, *, source_max_length) or None
        """

        assert len(source_sentences) == 1

        _, source_length, source_full_output, target_hidden = self(source_sentences)
        return self.decoder.beam_search(source_length, source_full_output, target_hidden, target_max_length, beam_size,
                                        need_attention=need_attention)
