import sys
sys.path.append('/root/metabci')
import os
import torch
from metabci.brainda.datasets import SEED
from metabci.brainda.paradigms import Emotion
import argparse
from metabci.brainda.algorithms.self_supervised_learning import EmoAdapt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, type=int)
    parser.add_argument('--fs', default=200, type=int)

    # Train Hyperparameter
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--train_epochs', default=300, type=int)
    parser.add_argument('--train_learning_rate', default=125e-6, type=float)
    parser.add_argument('--train_batch_size', default=64, type=int)

    # Model Hyperparameter
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--second', default=5, type=int)
    parser.add_argument('--time_window', default=1, type=int)
    parser.add_argument('--time_step', default=0.2, type=int)

    parser.add_argument('--encoder_embed_dim', default=768, type=int)
    parser.add_argument('--encoder_heads', default=8, type=int)
    parser.add_argument('--encoder_depths', default=4, type=int)
    parser.add_argument('--decoder_embed_dim', default=256, type=int)
    parser.add_argument('--decoder_heads', default=8, type=int)
    parser.add_argument('--decoder_depths', default=3, type=int) #3
    parser.add_argument('--alpha', default=1, type=float)

    parser.add_argument('--projection_hidden', default=[1024, 512], type=list)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--print_point', default=5, type=int)
    parser.add_argument('--ckpt_path', default=os.path.join('', 'models'), type=str)
    parser.add_argument('--model_name', default='EmoAdapt')
    return parser.parse_args()


##cross session
dataset_path = 'E:\SEED'
train_dataset = SEED(path=dataset_path, win_duration=5, sessions=[0, 1])
test_dataset = SEED(path=dataset_path, win_duration=5, sessions=[2])
paradigm = Emotion(
    srate=200,
    channels=["FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8"]
    # channels=["FP1", "C5", "CP3", "P4"]
)


X_train, y_train, meta_train = paradigm.get_data(
    train_dataset,
    subjects=[i+1 for i in range(15)],
    return_concat=True,
    n_jobs=5,
    verbose=False)

X_test, y_test, meta_test = paradigm.get_data(
    test_dataset,
    subjects=[i+1 for i in range(15)],
    return_concat=True,
    n_jobs=5,
    verbose=False)

args = get_args()

model = EmoAdapt(fs=args.fs, second=args.second, time_window=args.time_window, time_step=args.time_step,
                 encoder_embed_dim=args.encoder_embed_dim, encoder_heads=args.encoder_heads,
                 encoder_depths=args.encoder_depths,
                 decoder_embed_dim=args.decoder_embed_dim, decoder_heads=args.decoder_heads,
                 decoder_depths=args.decoder_depths,
                 channels=args.channels, train_batch_size=args.train_batch_size,
                 lr=args.train_learning_rate, train_epochs=args.train_epochs,
                 mask_ratio=args.mask_ratio, alpha=args.alpha,
                 print_point=args.print_point, device=device, random_seed=args.seed,
                 projection_hidden=args.projection_hidden, temperature=args.temperature,
                 model_path=args.ckpt_path, model_name=args.model_name, n_fold=args.n_fold)

# y_train and y_test is not used in training !!!!!
# only used for val and eval
model.fit(X_train, y_train, X_test, y_test)

