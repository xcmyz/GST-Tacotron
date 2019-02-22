import torch
import torch.nn as nn
from torch import optim

from Network import *
from gen_training_loader import DataLoader, collate_fn, SpeechData

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = nn.DataParallel(Tacotron()).to(device)
    print("Model Have Been Defined")

    # Get dataset
    dataset = SpeechData(args.dataset_path)
    # print(type(args.dataset_path))
    # print(len(dataset))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    # # Loss for frequency of human register
    # n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)

    # Get training loader
    print("Get Training Loader")
    training_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                 collate_fn=collate_fn, drop_last=True, num_workers=cpu_count())
    # print(len(training_loader))

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("---Model Restored at Step %d---\n" % args.restore_step)

    except:
        print("---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Training
    model = model.train()

    total_step = hp.epochs * len(training_loader)
    # print(total_step)
    Loss = []
    Time = np.array([])
    Start = time.clock()
    f = open("log.txt", "w")
    f.close()
    for epoch in range(hp.epochs):
        # print("########")
        for i,  data_batch in enumerate(training_loader):
            start_time = time.clock()
            # print("in")
            # Count step
            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1
            # print(current_step)

            # Init
            optimizer.zero_grad()

            #  {"text": texts, "mel": mels, "spec": specs}
            texts = data_batch["text"]
            # mels = trans(data_batch["mel"])
            # specs = trans(data_batch["spec"])
            mels = data_batch["mel"]
            specs = data_batch["spec"]

            mels_input = mels[:, :-1, :]  # (batch, lenght, num_mels)
            mels_input = mels_input[:, :, -hp.n_mels:]
            ref_mels = mels[:, 1:, :]
            # print(np.shape(mels))
            # print(np.shape(mels_input))
            # print(np.shape(ref_mels))

            if torch.cuda.is_available():
                texts = torch.from_numpy(texts).type(
                    torch.cuda.LongTensor).to(device)
            else:
                texts = torch.from_numpy(texts).type(
                    torch.LongTensor).to(device)
            mels = torch.from_numpy(mels).to(device)
            specs = torch.from_numpy(specs).to(device)
            mels_input = torch.from_numpy(mels_input).to(device)
            ref_mels = torch.from_numpy(ref_mels).to(device)
            # print(np.shape(specs))

            # Forward
            mel_output, linear_output, _ = model(texts, mels_input, ref_mels)
            # Attention: output: num_mel * r!!!
            # print("########################")
            # print(np.shape(mel_output))
            # print(np.shape(linear_output))

            # Loss
            # mel_loss = torch.abs(
            #     mel_output - compare(mel_output, mels[:, 1:, :], device))
            # mel_loss = torch.mean(mel_loss)
            # linear_loss = torch.abs(
            #     linear_output - compare(linear_output, specs[:, 1:, :], device))
            # linear_loss = torch.mean(linear_loss)
            # print(mel_output)
            mel_loss = torch.mean(torch.abs(mel_output - mels[:, 1:, :]))
            # print(mel_loss)
            linear_loss = torch.mean(
                torch.abs(linear_output - specs[:, 1:, :]))
            # print(linear_loss)
            loss = mel_loss + hp.loss_weight * linear_loss
            loss = loss.to(device)
            # print(current_step)
            # if current_step % hp.log_step == 0:
            #     print(loss)
            print(loss)
            Loss.append(loss)
            # et = time.clock()
            # print(et - st)
            # print(loss)

            # Backward
            loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), hp.clip_value)

            # Update weights
            optimizer.step()

            if current_step % hp.log_step == 0:
                Now = time.clock()
                # print("time per step: %.2f sec" % time_per_step)
                # print("At timestep %d" % current_step)
                # print("linear loss: %.4f" % linear_loss.data[0])
                # print("mel loss: %.4f" % mel_loss.data[0])
                # print("total loss: %.4f" % loss.data[0])
                str_loss = "Epoch [{}/{}], Step [{}/{}], Linear Loss: {:.4f}, Mel Loss: {:.4f}, Total Loss: {:.4f}.".format(
                    epoch+1, hp.epochs, current_step, total_step, linear_loss.item(), mel_loss.item(), loss.item())
                str_time = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now-Start), (total_step-current_step)*np.mean(Time))
                print(str_loss)
                print(str_time)
                with open("log.txt", "a") as f:
                    f.write(str_loss + "\n")
                    f.write(str_time + "\n")
                    f.write("\n")

            # print(current_step)
            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)

            if current_step in hp.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)

            end_time = time.clock()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)
                # print(Time)


def trans(arr):
    return np.stack([np.transpose(ele) for ele in arr])
    # for i, b in enumerate(arr):
    # arr[i] = np.transpose(b)


def adjust_learning_rate(optimizer, step):
    if step == 500000:
        # if step == 20:
        # print("update")
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


# def compare(out, stan, device):
#     # for batch_index in range(len(out)):
#     #     for i in range(min([np.shape(out)[2], np.shape(stan)[2]])):
#     #         torch.abs(out[batch_index][i], stan[batch_index][i])
#     # cnt = min([np.shape(out)[2], np.shape(stan)[2]])
#     if np.shape(stan)[2] >= np.shape(out)[2]:
#         return stan[:, :, :np.shape(out)[2]]
#     # return out[:,:,:cnt], stan[:,:,:cnt]
#     else:
#         frame_arr = np.zeros([np.shape(out)[0], np.shape(out)[1], np.shape(out)[
#                              2]-np.shape(stan)[2]], dtype=np.float32)
#         return torch.Tensor(np.concatenate((stan.cpu(), frame_arr), axis=2)).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='dataset path', default='dataset')
    # parser.add_argument('--restore_step', type=int,
    #                     help='Global step to restore checkpoint', default=0)
    # parser.add_argument('--batch_size', type=int,
    #                     help='Batch size', default=hp.batch_size)

    # Test
    parser.add_argument('--batch_size', type=int, help='Batch size', default=2)
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=0)

    args = parser.parse_args()
    main(args)
