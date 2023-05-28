from config.config_args import args
import torch
from models.lstm_models import LSTMModel


def load_model():
    # init model
    model = LSTMModel(args).to(args.device)
    # load_model version
    model.load_state_dict(torch.load(
        'checkpoints/mlp_best.pth', map_location=torch.device(args.device)))
    return model
