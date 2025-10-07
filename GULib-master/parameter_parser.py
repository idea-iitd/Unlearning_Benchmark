import argparse
import sys
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()
 
    #for all methods#
    parser.add_argument('--cuda', type=int, default=3, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--root_path', type=str, default='./', help='Set The Root Path')
    

    #data
    parser.add_argument('--dataset_name', type=str, default='cora',
                                          choices = ["cora",
                                                    "citeseer",
                                                    "pubmed",
                                                    "CS",
                                                    "Physics",
                                                    "flickr",
                                                    "Photo",
                                                    "Computers",
                                                    "DBLP",
                                                    "ogbl",
                                                    "ogbn-arxiv",
                                                    "Reddit",
                                                    "ogbn-products",
                                                    "Squirrel",
                                                    "Chameleon",
                                                    "Actor",
                                                    "Minesweeper",
                                                    "Tolokers",
                                                    "Amazon-ratings",
                                                    "Roman-empire",
                                                    "Questions",
                                                    "MUTAG",
                                                    "COX2",
                                                    "AIDS",
                                                    "BZR",
                                                    "DD",
                                                    "PROTEINS",
                                                    "ENZYMES",
                                                    "DHFR",
                                                    "NCI1",
                                                    "PTC_MR",
                                                    "ogbg-molhiv",
                                                    "ogbg-molpcba",
                                                    "ogbg-ppa",
                                                    "MNISTSuperpixels",
                                                    "ShapeNet",
                                                    "IMDB-BINARY",
                                                    "IMDB-MULTI"])
    parser.add_argument('--is_transductive', type=str2bool, default=True, help = "Task is transductive or inductive")
    parser.add_argument('--cal_mem', type=str2bool, default=False, help = "run exp to calculate memory")
    # parser.add_argument('--inductive', type=str, default='normal', choices=['cluster-gcn', 'graphsaint', 'normal'])
    parser.add_argument('--is_balanced' ,type = str2bool,default=False,help="dataset is split with balanced classes" )
    parser.add_argument('--use_batch', type=str2bool, default=False, help="train model with minibatch")
    parser.add_argument('--poison', type=str2bool, default=False, help="poisoned edge")
    parser.add_argument('--process', type=str, default="", help="process data",choices=["feature_noise", "feature_sparsity", "label_noise", "label_sparsity","None"])
    parser.add_argument('--noise_ratio', type=float, default=0.1, help="noise ratio")
    parser.add_argument('--sparsity_ratio', type=float, default=0.1, help="sparsity ratio")

    #modelMin
    parser.add_argument('--base_model', type=str, default='GCN', choices=["SIGN", "SGC","S2GC","SAGE", "GAT", 'Cluster_GCN', "GCN", "GIN",
                                                                          "GST","SAINT","Projector","Cheb","APPNP","GCN2","GATv2","TAG","LightGCN"])
    parser.add_argument('--unlearning_methods', type=str, default='SGU',
                        choices=['GraphEraser', 'GUIDE', 'GNNDelete', 'CEU', "GIF", "SGU","CGU","GST","Projector","MEGU","GraphRevoker","UTU","GUKD","D2DGN","IDEA","ScaleGUN"])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    #task
    parser.add_argument('--exp', type=str, default='sequence',
                        choices=["partition", "unlearning", "node_edge_unlearning", "attack_unlearning","sequence"])
    parser.add_argument('--unlearn_trainer', type=str, default='BaseTrainer')
    parser.add_argument('--parameter_task', type=str, default='normal', choices=['normal', "optuna"])
    parser.add_argument('--downstream_task', type=str, default='node', choices=['node', "edge","graph"])
    parser.add_argument('--unlearn_task', type=str, default='node', choices=['feature', "node", "edge"])

    #If running over copy unlearning data 
    parser.add_argument('--use_copy', type=str2bool, default=False, help = "True if Using different unleared set")
    
    #train#
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--opt_lr', type=float, default=0.001,help = "used for GraphEraser aggregating,GST and Projector and CGU")
    parser.add_argument('--opt_decay', type=float, default=0.0001,help = "used for GraphEraser aggregating,GST and Projector and CGU")
    parser.add_argument('--std', type=float, default=1e-2, help='standard deviation for objective perturbation for GST and CGU')
    parser.add_argument('--alpha', type=float, default=0.5,help='alpha in loss function for GNNDelete and CGU')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Choice of optimizer. [LBFGS/Adam] for GST and CGU')
    parser.add_argument('--lam', type=float, default=1e-2, help='L2 regularization for GST and CGU')
    parser.add_argument('--eps', type=float, default=1.0, help='Eps coefficient for certified removal for GST and CGU')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer for GST and CGU')
    parser.add_argument('--compare_gnorm', action='store_true', default=False,
                        help='Compute norm of worst case and real gradient each round for GST and CGU')
    
    #unlearning parameters#
    parser.add_argument('--num_unlearned_nodes', type=int, default=540)
    parser.add_argument('--num_unlearned_edges', type=int, default=1000)
    parser.add_argument('--proportion_unlearned_nodes', type=float, default=0.1)
    parser.add_argument('--proportion_unlearned_edges', type=float, default=0.1)
    parser.add_argument('--proportion_unlearned_edges_num', type=float, default=1e-4)
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)
    parser.add_argument('--unlearn_lr', type=float, default=0.01,help='used in GNNDelete and CEU')

    # MiA attack needed or not
    parser.add_argument('--attack', type=str2bool, default=False)

    #GUIDE parameter
    parser.add_argument('--GUIDE_methods', type=str, default= "Fast",choices=["Fast","SR"])
    parser.add_argument('--GUIDE_repair_methods', type=str, default= "MixUp",choices=["Zero", "Mirror", "MixUp", "NoneR"])


    #GraphEraser parameter
    parser.add_argument('--num_shards', type=int, default=10)
    parser.add_argument('--partition_method', type=str, default='lpa_base',
                        choices=["sage_km", "random", "lpa", "metis", "lpa_base", "sage_km_base","gpa","graph_km"])
    parser.add_argument('--opt_num_epochs', type=int, default=20)
    parser.add_argument('--ratio_deleted_edges', type=float, default=0)
    parser.add_argument('--aggregator', type=str, default='optimal', choices=['mean', 'majority', 'optimal','contrastive','kernel_similarity'])
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)
    parser.add_argument('--is_prune', type=str2bool, default=False)
    parser.add_argument('--is_partition', type=str2bool, default=True)
    parser.add_argument('--is_constrained', type=str2bool, default=True)
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_gen_embedding', type=str2bool, default=True)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--num_opt_samples', type=int, default=1000)

    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--repartition', type=str2bool, default=False)

    #GNNDelete parameter
    parser.add_argument('--checkpoint_dir', type=str, default= './data/GNNDelete/checkpoint_node',help='checkpoint folder')
    parser.add_argument('--random_seed', type=int, default=2024,help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=64,help='hidden dimension')
    parser.add_argument('--in_dim', type=int, default=128,help='input dimension')
    parser.add_argument('--out_dim', type=int, default=64,help='output dimension')
    parser.add_argument('--unlearning_model', type=str, default='gnndelete_nodeemb',help='unlearning method')
    parser.add_argument('--df', type=str, default='out',help='Df set to use')
    parser.add_argument('--df_idx', type=str, default='none',help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=0.5, help='Df size')
    
    parser.add_argument('--neg_sample_random', type=str, default='non_connected',help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean',help='loss function. one of {mse, kld, cosine}')
    parser.add_argument('--loss_type', type=str, default='both_layerwise',help='type of loss. one of {both_all, both_layerwise, only2_layerwise, only2_all, only1}')

    #CEU
    parser.add_argument('--train_batch', type=int, default=10)
    parser.add_argument('--test_batch', type=int, default=5)
    parser.add_argument('-l2', type=float, default=1E-5)
    parser.add_argument('--early_stop', type=str2bool, default=True)
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('--feature',type=str2bool,default=False,help='embedding feature')
    parser.add_argument('--feature_update', type=str2bool, default=True, help='embedding feature update')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding dim')
    parser.add_argument('-max-degree', action='store_true')
    parser.add_argument('-damping', type=float, default=0.)
    parser.add_argument('-hidden', type=int, nargs='+', default=[])
    parser.add_argument('-approx', type=str, default='cg')
    parser.add_argument('-depth', type=int, default=300)

    #GIF
    parser.add_argument('--GIF_method', type=str, default="GIF", choices=["GIF", "Retrain", "IF"])
    parser.add_argument('--GIF_exp', type=str, default='unlearning')
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--iteration', type=int, default=100)
    parser.add_argument('--scale', type=int, default=1000000000)
    parser.add_argument('--damp', type=float, default=0.0)


    #SGU
    parser.add_argument('--GNN_layer', type=int, default=3)
    parser.add_argument('--unlearning_epochs', type=int, default=50)
    parser.add_argument('--Budget', type=float, default=0.1)
    # parser.add_argument('--para1', type=float, default=0.01)
    # parser.add_argument('--para2', type=float, default=0.5)
    # parser.add_argument('--para3', type=float, default=10)
    # parser.add_argument('--para4', type=float, default=1.5)
    # parser.add_argument('--para5', type=float, default=1)
    parser.add_argument('--para1', type=float, default=2.5)
    parser.add_argument('--para2', type=float, default=0.01)
    parser.add_argument('--para3', type=float, default=250)
    parser.add_argument('--para4', type=float, default=0.1)
    parser.add_argument('--para5', type=float, default=10)


    #GST
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=10)
    parser.add_argument('--rm_disp_step', type=int, default=1)
    parser.add_argument('--J', type=int, default=5)
    parser.add_argument('--Q', type=int, default=4)
    parser.add_argument('--L', type=int, default=3)

    parser.add_argument('--remove_guo', action='store_true', default=False)
    parser.add_argument('--retrain', action='store_true', default=False, help='Retrain GST from scratch or not. If this is true then remove_guo should be false!')

    parser.add_argument('--GST_delta', type=float, default=1e-4, help='Delta coefficient for certified removal.')
   
    ###Projector
    parser.add_argument('--hop_neighbors', type=int, default=20)
    parser.add_argument("--dropout_times", type=int, default=2)
    parser.add_argument("--use_cross_entropy", action="store_true")
    parser.add_argument("--use_adapt_gcs", action="store_true")
    parser.add_argument("--x_iters", type=int, default=3)
    parser.add_argument("--y_iters", type=int, default=3)
    parser.add_argument('--require_linear_span', type=str2bool, default=True)
    parser.add_argument("--regen_model", type=str2bool, default=True)
    parser.add_argument("--parallel_unlearning", type=int, default=1)

    ###CGU
    parser.add_argument('--result_dir', type=str, default='./result/CGU',
                        help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='null', help='dataset')
    parser.add_argument('--train_mode', type=str, default='ovr', help='train mode [ovr/binary]')
    parser.add_argument('--train_sep', action='store_true', default=False,
                        help='train binary classifiers separately')
    # New arguments below
    parser.add_argument('--XdegNorm', type=str2bool, default=False, help='Apply our degree normaliztion trick')
    parser.add_argument('--add_self_loops', type=str2bool, default=True, help='Add self loops in propagation matrix')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay factor for Adam')
    parser.add_argument('--featNorm', type=str2bool, default=False, help='Row normalize feature to norm 1.')
    parser.add_argument('--GPR', action='store_true', default=False, help='Use GPR model')
    parser.add_argument('--balance_train', action='store_true', default=False,
                        help='Subsample training set to make it balance in class size.')
    parser.add_argument('--Y_binary', type=str, default='0',
                        help='In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).')
    parser.add_argument('--noise_mode', type=str, default='data',
                        help='Data dependent noise or worst case noise [data/worst].')
    parser.add_argument('--removal_mode', type=str, default='node', help='[feature/edge/node].')

    parser.add_argument('--delta', type=float, default=1e-4, help='Delta coefficient for certified removal.')
    parser.add_argument('--disp', type=int, default=5, help='Display frequency.')
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='Use fixed random seed for removal queue.')
    parser.add_argument('--compare_retrain', action='store_true', default=False,
                        help='Compare acc with retraining each round.')
    parser.add_argument('--compare_guo', action='store_true', default=False,
                        help='Compare performance with Guo et al.')
    
    ###MEGU###
    parser.add_argument('--kappa', type=float, default=0.01)
    parser.add_argument('--alpha1', type=float, default=0.8)
    parser.add_argument('--alpha2', type=float, default=0.5)

    ###GraphRevoker###
    parser.add_argument('--is_use_train_batch', type=str2bool, default=False)
    parser.add_argument('--is_use_test_batch', type=str2bool, default=False)


    ###IDEA###
    parser.add_argument('--unlearn_feature_partial_ratio', type=float, default=0.5)
    parser.add_argument('--gaussian_mean', type=float, default=0.0)
    parser.add_argument('--gaussian_std', type=float, default=0.0)
    parser.add_argument('--l', type=float, default=0.25, help="lipschitz constant of the loss.")
    parser.add_argument('--lambda', type=float, default=1.0, help="(original) loss function is lambda-strongly convex.")  # 0.05 1.0 
    parser.add_argument('--c', type=float, default=0.5, help="numerical bound of the training loss regarding each sample.")  # 3.0 0.5
    parser.add_argument('--M', type=float, default=0.25, help="the loss is M - Lipschitz Hessian in terms of w, i.e., gamma_1 in certified edge unlearning.")
    parser.add_argument('--c1', type=float, default=1.0, help="value of the derivative of loss is c1 bounded.")
    parser.add_argument('--lambda_edge_unlearn', type=float, default=1.0, help="regularization term weight - edge unlearning.")
    parser.add_argument('--gamma_2', type=float, default=1.0, help="lipschitz constant of first-order derivative of the loss - edge unlearning.")
    parser.add_argument('--file_name', type=str, default="unlearning_results", help="file name for results.")
    parser.add_argument('--write', type=bool, default=True, help="write to keep results.")

    ###UTULink###
    parser.add_argument("--eval_on_cpu", type=bool,default=False)

    ###ScaleGUN###
    parser.add_argument("--path", default="./data/ScaleGUN")
    parser.add_argument("--del_path_suffix", default="unlearning_data/")
    parser.add_argument("--analysis_path", default="analysis")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--axis_num", default=1, type=int, choices=[1, 0])
    parser.add_argument("--prop_algo", type=str,
                        choices=["power", "push", "MC"], default="MC")
    parser.add_argument("--prop_step", default=3, type=int)
    parser.add_argument("--r", default=0.5, type=float)
    parser.add_argument("--decay", default=0.1, type=float)
    parser.add_argument("--RW", type=int, default=20,
                        help="random walk times")
    parser.add_argument("--rmax", default=0.0, type=float)
    parser.add_argument("--ppr", default=False, action="store_true")
    parser.add_argument("--weight_mode", default="test",
                        type=str, choices=["decay", "avg", "test", "hetero"],)

    parser.add_argument("--optuna", action="store_true", default=False,
                        help="Use optuna to optimize hyperparameters.",)
    parser.add_argument("--del_postfix", type=str, default="")
    parser.add_argument("--del_only", default=False, action="store_true")
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--num_batch_removes", default=5, type=int)
    parser.add_argument("--no_retrain", action="store_true", default=True)
    parser.add_argument("--edge_idx_start", default=0, type=int)
    parser.add_argument("--num_removes", default=10, type=int,
                        help="number of removed edges/nodes in each batch",)
    
    parser.add_argument("--edge_del_file", type=str, default="",
                        help="Direct path to edge deletion file, overrides the default path construction")

    if "sphinx-build" in sys.argv[0] or os.environ.get('READTHEDOCS') == 'True':
        args = vars(parser.parse_args([]))
        return args
    else:
        args = vars(parser.parse_args())
        return args