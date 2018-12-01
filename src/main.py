##################################################
# Import modules

import random
import torch.optim.lr_scheduler

import matplotlib.pyplot as plt

import data
import metrics
import seq2seq
import train


##################################################
# Functions

def print_raw_sentences(name_list, raw_sentences_list):
    for i in range(len(raw_sentences_list[0])):
        for j in range(len(raw_sentences_list)):
            print('{} -> {}'.format(name_list[j], ' '.join(raw_sentences_list[j][i])))
        print()


def plot_loss(loss_list, loss_val_list):
    plt.plot(range(len(loss_list)), loss_list, label='training')
    plt.plot(range(len(loss_list)), loss_val_list, label='validating')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


##################################################
# Prepare data

en_lang, zh_lang = data.load_parallel_en_zh()

n_train = int(en_lang.n_sentences * 0.95)
n_val = en_lang.n_sentences - n_train

index = list(range(en_lang.n_sentences))
r = random.Random(0)
r.shuffle(index)

index_train = index[:n_train]
source_sentences = en_lang.to_sentences([en_lang.raw_sentences[i] for i in index_train], eos=False)
target_sentences = zh_lang.to_sentences([zh_lang.raw_sentences[i] for i in index_train], eos=True)

index_val = index[n_train:][:n_val]
source_sentences_val = en_lang.to_sentences([en_lang.raw_sentences[i] for i in index_val], eos=False)
target_sentences_val = zh_lang.to_sentences([zh_lang.raw_sentences[i] for i in index_val], eos=True)

# View data

print('{} training samples, {} validating samples'.format(len(source_sentences), len(source_sentences_val)))
print()

print_raw_sentences(
    ['en', 'zh'],
    [en_lang.to_raw_sentences(source_sentences[:5]),
     zh_lang.to_raw_sentences(target_sentences[:5])])

##################################################
# Overfit on small dataset

n_small = 100

model_s = seq2seq.Seq2seq('gru', 'bilinear', 1, en_lang.n_words, 100, 200, zh_lang.n_words, 100, 300)
optimizer_s = torch.optim.Adam(model_s.parameters(), lr=0.01)
scheduler_s = torch.optim.lr_scheduler.ExponentialLR(optimizer_s, 1.0)

# Train

loss_list_s, loss_val_list_s = train.train(
    model_s, optimizer_s, scheduler_s,
    source_sentences[:n_small], target_sentences[:n_small],
    source_sentences_val[:n_small], target_sentences_val[:n_small],
    teacher_force_rate=1.0, n_epochs=40, batch_size=500)

# Loss curve

plot_loss(loss_list_s, loss_val_list_s)

# Predict

target_sentences_g = train.predict_greedy(
    model_s, source_sentences[:n_small], target_max_length=10, batch_size=500)

# Show

print_raw_sentences(
    ['en', 'zh', 'greedy'],
    [en_lang.to_raw_sentences(source_sentences[:10]),
     zh_lang.to_raw_sentences(target_sentences[:10]),
     zh_lang.to_raw_sentences(target_sentences_g[:10])])

##################################################
# Run on full dataset

model = seq2seq.Seq2seq('gru', 'bilinear', 2, en_lang.n_words, 200, 400, zh_lang.n_words, 200, 600)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

# Train

loss_list, loss_val_list = train.train(
    model, optimizer, scheduler,
    source_sentences, target_sentences,
    source_sentences_val, target_sentences_val,
    teacher_force_rate=1.0, n_epochs=40, batch_size=500)

# Loss curve

plot_loss(loss_list, loss_val_list)

# Predict

target_sentences_val_g = train.predict_greedy(
    model, source_sentences_val, target_max_length=10, batch_size=500)
target_sentences_val_b = train.predict_beam(
    model, source_sentences_val[:1000], target_max_length=10, beam_size=10)

# Show

print_raw_sentences(
    ['en', 'zh', 'greedy', 'beam'],
    [en_lang.to_raw_sentences(source_sentences_val[:20]),
     zh_lang.to_raw_sentences(target_sentences_val[:20]),
     zh_lang.to_raw_sentences(target_sentences_val_g[:20]),
     zh_lang.to_raw_sentences(target_sentences_val_b[:20])])

# BLEU score

raw_target_sentences_val = zh_lang.to_raw_sentences(target_sentences_val)
raw_target_sentences_val_g = zh_lang.to_raw_sentences(target_sentences_val_g)
raw_target_sentences_val_b = zh_lang.to_raw_sentences(target_sentences_val_b)

bleu_g = metrics.bleu(raw_target_sentences_val_g, raw_target_sentences_val)
bleu_b = metrics.bleu(raw_target_sentences_val_b, raw_target_sentences_val)

print('bleu score (greedy): {}, bleu score (beam): {}'.format(bleu_g, bleu_b))
print()

# Attention

##################################################
