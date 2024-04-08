import torch
import torch.nn as nn
import ultralytics

def load_available_state_dict(model: nn.Module, pth: str):
    """从pth中加载model里存在的部分

    Args:
        model (nn.Module): 要加载的模型
        pth (str): 待加载权重路径

    Returns:
        nn.Module: 加载权重后的模型
    """
    weight_dict = torch.load(pth)
    model_dict = model.state_dict()
    new_dict = {k:v for k, v in weight_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    return model

# 先看model1均为0，然后加载，变成1
# model_1 = torch.load('test.pth')
# backbone_1 = torch.load('backbone.pth')
# model.load_state_dict(model_1)
# model = load_available_state_dict(model, 'backbone.pth')
# print(model.state_dict())
# print(backbone.state_dict())

""" """""""""""""""""""""""""""""""""""""""""""""""""""""""""""" """
import re

def offset(type: str) -> int:
    """
    定义了Fusion模型中RGB的backbone和IR的backbone位置偏移
    """
    if type == "RGB":
        return 3
    if type == "T":
        return 13
    return 0

def replace_first_num(s: str, num: int) -> str:
    """
    将字符串s中的第一个数字替换为num, 然后返回新的字符串
    """
    match = re.search(r'\d+', s)
    if match:
        start_index = match.start()
        end_index = match.end()
        new_s = s[:start_index] + str(num) + s[end_index:]
    else:
        new_s = s
    return new_s

def get_backbone_dict(module: nn.Module, type: str, backbone: range) -> dict:
    """
    从module中获取backbone的权重字典, 将字典中的权重key根据模态替换为Fusion中的key, 返回新的字典
    Args:
      module: 模型
      type: 模态
      backbone: backbone所处的层
    """
    assert type in ('RGB', 'T'), f"Unable to recognize the type content"
    
    backbone_dict = {}
    for k, v in module["model"].state_dict().items():
        if k == 'model.0.conv.weight': # 第一层由于channel大小不同，无法替换
            continue
        layer = int(k.split('.')[1])
        if layer in backbone:
            new_layer = layer + offset(type)
            k = replace_first_num(k, new_layer)
            backbone_dict[k] = v
    return backbone_dict

def load_backbone(rgbt_module: nn.Module, rgb_module: str, t_module: str):
    """
    替换rgbt_module中RGB和IR部分的backbone
    """
    rgb_module_dict = torch.load(rgb_module)
    t_module_dict = torch.load(t_module)
    
    rgb_backbone_dict = get_backbone_dict(rgb_module_dict, 'RGB', range(0, 10))
    t_backbone_dict = get_backbone_dict(t_module_dict, 'T', range(0, 10))
    
    update_dict = rgbt_module["model"].state_dict()
    update_dict.update(rgb_backbone_dict)
    update_dict.update(t_backbone_dict)
    rgbt_module["model"].load_state_dict(update_dict, strict=False)
    
    return rgbt_module

def check_backbone(rgbt_module: nn.Module, rgb_module: nn.Module, type:str):
    """
    判断Fusion模型中的backbone和单模态的backbone一不一样
    """
    for k, v in rgb_module["model"].state_dict().items():
        k_list = k.split('.')
        layer_id = int(k_list[1])
        if layer_id in range(0, 10):
            rgbt_layer_id = layer_id + offset(type)
            k_list[1] = str(rgbt_layer_id)
            rgbt_layer_key = ".".join(k_list)
            try:
                rgb_v = rgbt_module["model"].state_dict()[rgbt_layer_key]
                # print(k, rgb_layer_key, rgb_v.equal(v.cpu()))
                print(f"{k:<30} {rgbt_layer_key:<30} {rgb_v.cpu().equal(v.cpu())}")
            except KeyError as e:
                break
        

def check_if_the_backbone_weight_is_replace():
    """
    检查权重是否替换成功
    """
    rgb_model = torch.load("rgb.pt")
    ir_model = torch.load("ir.pt")
    rgbt_model = torch.load("rgbt-update.pt")
    check_backbone(rgbt_model, rgb_model, "RGB")
    check_backbone(rgbt_model, ir_model, "T")
    
def replace_weight_in_rgbt_module():
    """
    替换权重
    """
    rgbt_model = torch.load("rgbt.pt")
    rgbt_model = load_backbone(rgbt_model, "rgb.pt", "ir.pt")
    torch.save(rgbt_model, "rgbt-update.pt")

# replace_weight_in_rgbt_module()
# check_if_the_backbone_weight_is_replace()