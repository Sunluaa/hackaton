from random import seed

import numpy as np

from catboost import CatBoostClassifier
from catboost import CatBoostRanker
from lightgbm import LGBMClassifier
from lightgbm import LGBMRanker
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class Config:
    # ====== ======    ====== ======    ====== ======    ====== ======    ====== ======    ====== ======    ====== ======    ====== ======    ====== ======    
    # General
    # ============    ============    ============    ============    ============    ============    ============    ============    ============      
    name = "1_create_features"
    seed = 42
    is_train = False # True False
    is_classificate = True

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Path and submission
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    path_to_train = "data/train_dataset_small.pq"
    path_to_test = "data/test_dataset_small.pq"
    path_to_test_features = "data/features_small.pq"

    path_to_save_data_checkpoint = "./mlpipeline/artifacts/checkpoints/data_checkpoint.pkl"
    path_to_load_data_checkpoint = "./mlpipeline/artifacts/checkpoints/data_checkpoint.pkl"

    path_to_submission_dir = "./mlpipeline/artifacts/submissions"

    path_to_save_models = "./mlpipeline/artifacts/models"
    path_to_load_models = "./mlpipeline/artifacts/models"

    path_to_save_ensemble = "./mlpipeline/artifacts/ensembles/ensemble.pkl"
    path_to_load_ensemble = "./mlpipeline/artifacts/ensembles/ensemble.pkl"

    submission_id_column = "request_id"
    submission_target_column = "score"

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Metric & Loss
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    column_target = "is_deal"
    group_column = "request_id"
    metric_name = "ndcg"
    metric_at_k = 5
    prob_thrashold = 0.5


    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Preprocessing
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    force_categorical_columns = []


    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Validation
    # ============    ============    ============    ============    ============    ============    ============    ============    ============
    is_holdout = True
    val_size = 0.1
    test_shuffle = True

    is_cv = True
    cv_n_splits = 5
    cv_shuffle = True
    is_stratify = True

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Modeling
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    model = {
        'CatBoostYetiRank':         False,
        'LambdaMART':               True,
    }

    if True:
        catboost_yetirank_params = {
            'loss_function': 'YetiRank',
            'eval_metric': 'NDCG:top=5',
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'random_strength': 0.5,
            'border_count': 128,
            'random_seed': seed,
            'verbose': False,
        }


        lambdamart_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg@5',
            'learning_rate': 0.05,
            'n_estimators': 200,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': seed,
            'n_jobs': -1,
            'verbosity': -1,
        }
    
    models = {
        'CatBoostYetiRank': CatBoostRanker(**catboost_yetirank_params),
        'LambdaMART': LGBMRanker(**lambdamart_params),
    }

    fit_params = {
        'CatBoostYetiRank': {'cat_features': '__auto__'},
        'LambdaMART': {},
    }

    preprocessing_profiles = {
        "linear": {
            "scale_num": True,
            "encode_cat": "onehot",
        },
        "tree": {
            "scale_num": False,
            "encode_cat": "ordinal",
        },
        "catboost": {
            "scale_num": True,
            "encode_cat": "none",
        },
    }

    onehot_max_cardinality = 100
    high_cardinality_strategy = "frequency"

    model_preprocess_profile = {
        'CatBoostYetiRank':         "catboost",
        'LambdaMART':               "tree",
    }


    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Feature engineering
    # ============    ============    ============    ============    ============    ============    ============    ============    ============  
    drop_columns = [
        "request_id",
        "app_id",
        "request_received",
        "date_part",
        "offer_id",
    ] + [
        # 'term', 
        #  'req_loan_amount',
        #  'need_2ndfl', 
        #  'req_term',
        #  'rate',
        #  'verif_need',
        #  'verif_compl',
    ]


    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Ensebling
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    threshold = 0.5

    ensemble_method = {
        'mean': True,
        'vote': False,
        'linear': False,
        'ridge': False,
    }
    ensemble_alpha = 1.0

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Feature Selection
    # ============    ============    ============    ============    ============    ============    ============    ============    ============  

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Work on dataset?
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Baseline?
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     

    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Training?
    # ============    ============    ============    ============    ============    ============    ============    ============    ============
    blending_method = 'mean'
    # ============    ============    ============    ============    ============    ============    ============    ============    ============     
    # Logs
    # ============    ============    ============    ============    ============    ============    ============    ============    ============
     
    is_console_log = True      
