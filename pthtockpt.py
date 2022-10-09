import torch
import mindspore

dict_list = []
# for s1, s2 in zip(torch.load('rexnetv1_100-1b4dddf4.pth').items(), mindspore.load_checkpoint('rexnet.ckpt').items()):
    # if s1[0] != s2[0]:
    #     print(s1[0], s2[0])
    # exit()
    # print(s1[0])
for name, value in torch.load('D:\\AboutStudy\\ConViT\\源代码\\注释版\\convit_tiny.pth', map_location="cpu").items():
    param_dict = {}
    if name.endswith('norm1.weight'):
        name = name.replace('norm1.weight', 'norm1.gamma')
    elif name.endswith('norm2.weight'):
        name = name.replace('norm2.weight', 'norm2.gamma')
    elif name.endswith('norm1.bias'):
        name = name.replace('norm1.bias', 'norm1.beta')
    elif name.endswith('norm2.bias'):
        name = name.replace('norm2.bias', 'norm2.beta')
    elif name.endswith('norm.weight'):
        name = name.replace('norm.weight', 'norm.gamma')
    elif name.endswith('norm.bias'):
        name = name.replace('norm.bias', 'norm.beta')
    elif name.endswith('head.weight'):
        name = name.replace('head.weight', 'classifier.weight')
    elif name.endswith('head.bias'):
        name = name.replace('head.bias', 'classifier.bias')
    param_dict['name'] = name
    param_dict['data'] = mindspore.Tensor(value.numpy(), mindspore.float32)
    dict_list.append(param_dict)
    
mindspore.save_checkpoint(dict_list, 'convit.ckpt')
# for name, value in mindspore.load_checkpoint('rexnet.ckpt').items():
#     print(name)
# print(len(torch.load('rexnetv1_100-1b4dddf4.pth').items()), len(mindspore.load_checkpoint('rexnet.ckpt').items()))