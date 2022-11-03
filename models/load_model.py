from config.config_args import args
import torch
from models.gru_model import GRUModel

def load_model():
    ## init model
    model = GRUModel(args).to(args.device)
    #load_model version
    model.load_state_dict(torch.load(f = f'checkpoints/{args.version}_best.pth', map_location = torch.device(args.device)))
    return model