import torch
import os

def load_params(model_type, model_path, model=None, run_number=0, num_runs=1):
    if model_type in ["GIF", "IDEA"]:
        params_esti = torch.load(model_path, map_location="cpu")
        flat_params = torch.cat([p.flatten() for p in params_esti])
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
        flat_params = torch.cat([p.flatten() for p in state_dict.values()])
    return flat_params
def l2_distance(p1, p2):
    return torch.norm(p1 - p2).item()

def cosine_similarity(p1, p2):
    return torch.nn.functional.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()

def relative_l2(p1, p2):
    return (torch.norm(p1 - p2) / torch.norm(p1)).item()

datas = ["cora","citeseer","Photo","Amazon-ratings", "Roman-empire","ogbn-arxiv"]
unlearn_task = "node"
unlearn_ratio = "ratio_0.80"

for dataset in datas:
    # === Paths ===
    original_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/data/model/node_level/{dataset}/{unlearn_task}/GCN"

    results = {"GOLD_vs_Original": { "Rel-L2": []},
            "GOLD_vs_GIF":      { "Rel-L2": []},
            "GOLD_vs_IDEA":     { "Rel-L2": []},
            "GOLD_vs_MEGU":     { "Rel-L2": []}}

    for run in range(5):
        gold_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GOLD/{dataset}/{unlearn_task}/{unlearn_ratio}/GOLD_{dataset}_node_{unlearn_ratio}_{str(run)}.pt"
        GIF_model_path  = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/GIF/{dataset}/{unlearn_task}/{unlearn_ratio}/GIF_{dataset}_node_{unlearn_ratio}_{str(run)}.pt"
        IDEA_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/IDEA/{dataset}/{unlearn_task}/{unlearn_ratio}/IDEA_{dataset}_node_{unlearn_ratio}_{str(run)}.pt"
        MEGU_model_path = f"/DATATWO/users/gcond/data/unlearning/GNNDelete/ScaleGUN/GU_benchmark/OpenGU/unlearned_models/MEGU/{dataset}/{unlearn_task}/{unlearn_ratio}/MEGU_{dataset}_node_{unlearn_ratio}_{str(run)}.pt"

        params_original = load_params("GOLD", original_model_path)
        params_gold     = load_params("GOLD", gold_model_path)
        params_gif      = load_params("GIF",  GIF_model_path)
        params_idea     = load_params("IDEA", IDEA_model_path)
        params_megu     = load_params("MEGU", MEGU_model_path)

        # Compute metrics
        # results["GOLD_vs_Original"]["L2"].append(l2_distance(params_gold, params_original))
        # results["GOLD_vs_Original"]["Cosine"].append(cosine_similarity(params_gold, params_original))
        results["GOLD_vs_Original"]["Rel-L2"].append(relative_l2(params_gold, params_original))

        # results["GOLD_vs_GIF"]["L2"].append(l2_distance(params_gold, params_gif))
        # results["GOLD_vs_GIF"]["Cosine"].append(cosine_similarity(params_gold, params_gif))
        results["GOLD_vs_GIF"]["Rel-L2"].append(relative_l2(params_gold, params_gif))

        # results["GOLD_vs_IDEA"]["L2"].append(l2_distance(params_gold, params_idea))
        # results["GOLD_vs_IDEA"]["Cosine"].append(cosine_similarity(params_gold, params_idea))
        results["GOLD_vs_IDEA"]["Rel-L2"].append(relative_l2(params_gold, params_idea))

        # results["GOLD_vs_MEGU"]["L2"].append(l2_distance(params_gold, params_megu))
        # results["GOLD_vs_MEGU"]["Cosine"].append(cosine_similarity(params_gold, params_megu))
        results["GOLD_vs_MEGU"]["Rel-L2"].append(relative_l2(params_gold, params_megu))

    # === Print averages and std ===
    print("\n=== Averages & Standard Deviations over Runs ===")
    for key, metrics in results.items():
        for metric_name, values in metrics.items():
            t = torch.tensor(values)
            print(f"{dataset} {key} {metric_name:8s} - Mean: {t.mean().item()}, Std: {t.std().item()}")
