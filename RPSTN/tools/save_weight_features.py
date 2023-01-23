import torch
import numpy as np
import _init_paths
import models

from core.config import config


def load_pretrained_model(pretrained, model, dataset='Penn_Action'):
    if pretrained:
        checkpoint = torch.load(pretrained)
        p = checkpoint['state_dict']
        # print(p.keys())

        if dataset == "Penn_Action":
            prefix = 'invalid'

        state_dict = model.state_dict()
        model_dict = {}

        for k,v in p.items():
            if k in state_dict:
                # print(k.startswith(prefix))
                if not k.startswith(prefix):                                
                    model_dict[k] = v

        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        print('Loading Successfully')
        return model


pretrained = 'experiments/checkpoint/non-local/13-kps-res18-nl-current-frame_20201112_best.pth.tar'
SPTNet = models.dkd_net.get_dkd_net(config, is_train=False)
SPTNet = load_pretrained_model(pretrained, SPTNet)

for name, parameters in SPTNet.named_parameters():
    print(name, ': ', parameters.size())