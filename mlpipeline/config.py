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
    name = "1_add_stratified_group_kfold_cv"
    seed = 42
    is_train = True # True False
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
        'RandomForest':             False,
        'LambdaMART':               True,
    }

    if True:
        random_forest_params = {
            'n_estimators': 200,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'max_depth': 6,
            'criterion': 'log_loss',
            'class_weight': None,
            'ccp_alpha': 0.005,
            'bootstrap': True,
            "random_state": seed,
        }

        catboost_params = {
            'random_strength': 0.5,
            'learning_rate': 0.05,
            'l2_leaf_reg': 2,
            'iterations': 500,
            'depth': 3,
            'border_count': 128,
            'bagging_temperature': 0,
            "random_seed": seed,
            "verbose": False,
        }

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

        lgbm_params = {
            'verbosity': -1, 
            'subsample_freq': 1, 
            'subsample': 1.0, 
            'reg_lambda': 1.0, 
            'reg_alpha': 0.0, 
            'random_state': 42, 
            'objective': 'binary', 
            'num_leaves': 31, 
            'n_jobs': -1, 
            'n_estimators': 100, 
            'min_split_gain': 0.001, 
            'min_child_weight': 1.0, 
            'min_child_samples': 20, 
            'max_depth': -1, 
            'learning_rate': 0.1, 
            'colsample_bytree': 0.8, 
            'class_weight': 'balanced', 
            'boosting_type': 'dart'
        }

        lambdamart_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
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

        xgb_params = {
            'subsample': 0.8,
            'reg_lambda': 0.5,
            'reg_alpha': 0.0,
            'n_estimators': 200,
            'min_child_weight': 1,
            'max_depth': 2,
            'learning_rate': 0.1,
            'gamma': 0.01,
            'colsample_bytree': 1.0,
            "random_state": seed,
            "eval_metric": "logloss",
        }
    
    models = {
        'CatBoostYetiRank': CatBoostRanker(**catboost_yetirank_params),
        'RandomForest': RandomForestClassifier(**random_forest_params),
        'CatBoost': CatBoostClassifier(**catboost_params),
        'LightGBM': LGBMClassifier(**lgbm_params),
        'LambdaMART': LGBMRanker(**lambdamart_params),
        'XGBoost': XGBClassifier(**xgb_params),
    }

    fit_params = {
        'LogisticRegression_Lasso': {},
        'LogisticRegression_Ridge': {},
        'LogisticRegression_Elic': {},
        'KNN': {},
        'DecisionTree': {},
        'CatBoostYetiRank': {'cat_features': '__auto__'},
        'RandomForest': {},
        "CatBoost": {'cat_features': '__auto__',},
        'LightGBM':  {},
        'LambdaMART': {},
        'XGBoost':  {}, 
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
        'LogisticRegression_Lasso': "linear",
        'LogisticRegression_Ridge': "linear",
        'LogisticRegression_Elic':  "linear",
        'KNN':                      "linear",
        'DecisionTree':             "tree",
        'CatBoostYetiRank':         "catboost",
        'RandomForest':             "tree",
        "CatBoost":                 "catboost",
        'LightGBM':                 "tree",
        'LambdaMART':               "tree",
        'XGBoost':                  "tree", 
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
