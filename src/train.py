import time
import torch

__all__ = ['train', 'predict_greedy', 'predict_beam']


class Dataset:
    def __init__(self, *sentences_tuple, batch_size=1, shuffle=False):
        n_sentences = len(sentences_tuple[0])
        assert all(len(sentences) == n_sentences for sentences in sentences_tuple)

        self.sentences_tuple = sentences_tuple
        self.batch_size = batch_size
        self.n_batchs = (n_sentences + batch_size - 1) // batch_size
        self.n_sentences = n_sentences
        self.shuffle = shuffle

    def __iter__(self):
        batch_size = self.batch_size
        index = torch.randperm(self.n_sentences) if self.shuffle else range(self.n_sentences)
        for batch in range(self.n_batchs):
            index_batch = index[batch * batch_size: (batch + 1) * batch_size]
            yield tuple([sentences[i] for i in index_batch] for sentences in self.sentences_tuple)


def train_epoch(model, optimizer, dataset, teacher_force_rate, print_every=100):
    model.train()
    average_loss = 0.0
    total_loss = 0.0
    total_number = 0
    t0 = time.time()
    for batch, (source_sentences_batch, target_sentences_batch) in enumerate(dataset):
        current_number = len(source_sentences_batch)
        teacher_force = bool(torch.rand(()) < teacher_force_rate)
        optimizer.zero_grad()
        loss = model.cal_loss(source_sentences_batch, target_sentences_batch, teacher_force=teacher_force)
        (loss / current_number).backward()
        optimizer.step()
        average_loss = 0.9 * average_loss + 0.1 * (loss.item() / current_number)
        total_loss += loss.item()
        total_number += current_number
        if (batch + 1) % print_every == 0 or (batch + 1) == dataset.n_batchs:
            t1 = time.time()
            print('train_epoch ---- batch: ({}/{}), sentence: ({}/{}),'
                  ' average_loss: {:.3g}, loss: {:.3g}, time: {:.3f}s'.format(
                batch + 1, dataset.n_batchs, total_number, dataset.n_sentences,
                average_loss / (1 - 0.9 ** (batch + 1)), total_loss / total_number, t1 - t0))
    return total_loss / total_number


def evaluate_epoch(model, dataset, teacher_force_rate, print_every=100):
    model.eval()
    total_loss = 0.0
    total_number = 0
    t0 = time.time()
    for batch, (source_sentences_batch, target_sentences_batch) in enumerate(dataset):
        current_number = len(source_sentences_batch)
        teacher_force = bool(torch.rand(()) < teacher_force_rate)
        with torch.no_grad():
            loss = model.cal_loss(source_sentences_batch, target_sentences_batch, teacher_force=teacher_force)
        total_loss += loss.item()
        total_number += current_number
        if (batch + 1) % print_every == 0 or (batch + 1) == dataset.n_batchs:
            t1 = time.time()
            print('evaluate_epoch ---- batch: ({}/{}), sentence: ({}/{}), loss: {:.3g}, time: {:.3f}s'.format(
                batch + 1, dataset.n_batchs, total_number, dataset.n_sentences, total_loss / total_number, t1 - t0))
    return total_loss / total_number


def train(model, optimizer, scheduler, source_sentences, target_sentences, source_sentences_val, target_sentences_val,
          teacher_force_rate, n_epochs, batch_size, print_every=100):
    dataset = Dataset(source_sentences, target_sentences, batch_size=batch_size, shuffle=True)
    dataset_val = Dataset(source_sentences_val, target_sentences_val, batch_size=batch_size, shuffle=True)
    loss_list = []
    loss_val_list = []
    t0 = time.time()
    for epoch in range(n_epochs):
        loss = train_epoch(model, optimizer, dataset, teacher_force_rate, print_every=print_every)
        loss_val = evaluate_epoch(model, dataset_val, teacher_force_rate, print_every=print_every)
        scheduler.step()
        loss_list.append(loss)
        loss_val_list.append(loss_val)
        t1 = time.time()
        print('train ---- epoch: ({}/{}), loss: {:.3g}, loss_val: {:.3g}, time: {:.3f}'.format(
            epoch + 1, n_epochs, loss, loss_val, t1 - t0))
        print()
    return loss_list, loss_val_list


def predict_greedy(model, source_sentences, target_max_length, batch_size, print_every=100):
    model.eval()
    dataset = Dataset(source_sentences, batch_size=batch_size, shuffle=False)
    target_sentences_o = []
    t0 = time.time()
    for batch, (source_sentences_batch,) in enumerate(dataset):
        with torch.no_grad():
            target_sentences_o_batch, _ = model.greedy_search(
                source_sentences_batch, target_max_length, need_attention=False)
        target_sentences_o.extend(target_sentences_o_batch)
        if (batch + 1) % print_every == 0 or (batch + 1) == dataset.n_batchs:
            t1 = time.time()
            print('predict_greedy ---- batch: ({}/{}), sentence: ({}/{}), time: {:.3f}s'.format(
                batch + 1, dataset.n_batchs, len(target_sentences_o), dataset.n_sentences, t1 - t0))
    print()
    return target_sentences_o


def predict_beam(model, source_sentences, target_max_length, beam_size, print_every=100):
    model.eval()
    dataset = Dataset(source_sentences, batch_size=1, shuffle=False)
    target_sentences_o = []
    t0 = time.time()
    for batch, (source_sentences_batch,) in enumerate(dataset):
        with torch.no_grad():
            target_sentences_o_batch, _ = model.beam_search(
                source_sentences_batch, target_max_length, beam_size, need_attention=False)
        target_sentences_o.append(target_sentences_o_batch[0])
        if (batch + 1) % print_every == 0 or (batch + 1) == dataset.n_batchs:
            t1 = time.time()
            print('predict_beam ---- batch: ({}/{}), sentence: ({}/{}), time: {:.3f}s'.format(
                batch + 1, dataset.n_batchs, len(target_sentences_o), dataset.n_sentences, t1 - t0))
    print()
    return target_sentences_o
