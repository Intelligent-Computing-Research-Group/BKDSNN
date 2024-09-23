import torch

if __name__ == '__main__':
    path = "/home/zekai_xu/SNN/code/Spike-Element-Wise-ResNet/imagenet/sew34_checkpoint_319.pth"
    sdict = torch.load(path)['model']
    sdict_new = {}
    for k, v in sdict.items():
        sdict_new[k.replace("module.", "")] = v
    outdict = torch.load(path)
    outdict["model"] = sdict_new
    torch.save(outdict, "/home/zekai_xu/SNN/code/Spike-Element-Wise-ResNet/imagenet/best.pth")