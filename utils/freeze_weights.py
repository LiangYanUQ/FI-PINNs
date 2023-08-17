from collections.abc import Iterable
 
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)
 
def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)
 
def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children())[0])
    idxs = [2 * idx for idx in idxs]
    if idxs[-1] > num_child:
        raise ValueError("The demand is beyond the number of linear fields") 
    for idx, child in enumerate(list(model.children())[0]):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)
 
def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)