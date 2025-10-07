import torch

@torch.no_grad()
def member_infer_attack(target_model, attack_model, data, logits=None):
    '''Membership inference attack'''

    edge = data.train_pos_edge_index[:, data.df_mask]
    z = target_model(data.x, data.train_pos_edge_index[:, data.dr_mask])
    feature1 = target_model.decode(z, edge).sigmoid()
    feature0 = 1 - feature1
    feature = torch.stack([feature0, feature1], dim=1)
    # feature = torch.cat([z[edge[0]], z[edge][1]], dim=-1)
    logits = attack_model(feature)
    _, pred = torch.max(logits, 1)
    suc_rate = 1 - pred.float().mean()

    return torch.softmax(logits, dim=-1).squeeze().tolist(), suc_rate.cpu().item()