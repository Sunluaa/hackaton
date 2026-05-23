import os
import pickle
import numpy as np

import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class DataProcess:
    ''' По задумке тут происходят вся обработка датасета.'''

    def __init__(self, cfg, model_name) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.data_checkpoint = {}
        self._reported_constant_columns = set()

    # === ====== ===
    # Главное api класса
    # === ====== ===
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit(self, df):
        X, _ = self.split_Xy(df)
        self._fit_pipeline(X)
        self.save_data_checkpoint()
 
    def transform(self, df):
        if not self.data_checkpoint:
            raise ValueError(
                "data_checkpoint is empty. Pass the fitted checkpoint explicitly "
                "or call fit/fit_transform on this instance first."
            )

        X, y = self.split_Xy(df)
        X = self._transform_pipeline(X)

        if y is not None:
            return pd.concat([X, y], axis=1)
        return X

    # === ====== ===   
    # Дополнительные функции api класса
    # === ====== ===
    def split_Xy(self, df):
        y = df[self.cfg.column_target] if self.cfg.column_target in df.columns else None
        if y is not None:
            X = df.drop(columns=[self.cfg.column_target])
        else: 
            X = df.copy(deep=False)
        return X, y

    def cv_splits(self, df):
        n_splits = getattr(self.cfg, "cv_n_splits", 5)
        shuffle = getattr(self.cfg, "cv_shuffle", True)
        random_state = getattr(self.cfg, "seed", 42)
        group_col = getattr(self.cfg, "group_column", None)

        X, y = self.split_Xy(df)
        if y is None:
            raise ValueError("Target column is required for cv_splits.")

        if group_col and group_col in X.columns:
            groups = X[group_col]

            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
            )

            return list(splitter.split(X, y, groups))

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

        return list(skf.split(X, y))

    # === ====== ===
    # Пайплайн обучения и трансформации
    # === ====== ===
    def _fit_pipeline(self, X):
        temp = X.copy(deep=False)
        preprocess_cfg = self._resolve_preprocess_cfg()
        num_cols, cat_cols = self._infer_feature_types(temp)

        temp, num_cols, cat_cols = self._feature_engineering_before_na(temp, num_cols, cat_cols)
        temp, na_artifacts = self._na_processing(temp, num_cols, cat_cols, is_transform=False)
        temp, num_cols, cat_cols = self._feature_engineering_after_na(temp, num_cols, cat_cols)
        temp, drop_columns, constant_columns, num_cols, cat_cols = self._drop(temp, num_cols, cat_cols)
        temp, num_cols, cat_cols, pp_artifacts = self._preprocessing(temp, num_cols, cat_cols, preprocess_cfg, is_transform=False)

        temp, num_cols, cat_cols = self._standardization_df(temp, num_cols, cat_cols)

        self.data_checkpoint = {
            "experiment_name": getattr(self.cfg, "name", "default_experiment"),
            "seed": getattr(self.cfg, "seed", 42),
            "model_name": self.model_name,
            "preprocess_cfg": dict(preprocess_cfg),
            "num_cols": list(num_cols),
            "cat_cols": list(cat_cols),
            "final_columns": list(temp.columns),
            "na_artifacts": na_artifacts,
            "pp_artifacts": pp_artifacts,
            "drop_columns": drop_columns,
            "constant_columns": constant_columns,
        }

    def _transform_pipeline(self, X):
        if not self.data_checkpoint:
            raise ValueError("data_checkpoint is empty. Fit the processor or load checkpoint first.")

        preprocess_cfg = self._resolve_preprocess_cfg()

        num_cols, cat_cols = self._infer_feature_types(X)
        X, num_cols, cat_cols = self._feature_engineering_before_na(X, num_cols, cat_cols)
        X, _ = self._na_processing(X, num_cols, cat_cols, is_transform=True)
        X, num_cols, cat_cols = self._feature_engineering_after_na(X, num_cols, cat_cols)
        X, _, _, num_cols, cat_cols = self._drop(X, num_cols, cat_cols)
        X, num_cols, cat_cols, _ = self._preprocessing(X, num_cols, cat_cols, preprocess_cfg, is_transform=True)
        X, num_cols, cat_cols = self._standardization_df(X, num_cols, cat_cols)

        final_columns = self.data_checkpoint.get("final_columns")
        if final_columns is None:
            raise ValueError("Checkpoint does not contain final_columns.")

        X = X.reindex(columns=final_columns, fill_value=0.0)
        return X 

    # === ====== ===
    # Обработка датасета
    # === ====== ===
    def _na_processing(self, X, num_cols, cat_cols, is_transform):
        num_cols = [col for col in num_cols if col in X.columns]
        cat_cols = [col for col in cat_cols if col in X.columns]

        if not is_transform:
            # embarked_mode = X["Embarked"].mode(dropna=True)
            fill_value = {
                'basket_name': 'None',
            }
            # fill_value = {
            #     'Age_by_initial': X.groupby('Initial')['Age'].median().to_dict(),
            #     'Age': X['Age'].median(),
            #     'Embarked': embarked_mode.iloc[0],
            #     'Fare': X["Fare"].median()
            # } 

            num_cols = [col for col in num_cols if col not in fill_value]
            cat_cols = [col for col in cat_cols if col not in fill_value]
            
            cat_fill_values = {}
            for col in cat_cols:
                mode = X[col].mode(dropna=True)
                cat_fill_values[col] = str(mode.iloc[0]) if not mode.empty else "missing"

            num_fill_values = {}
            for col in num_cols:
                median = X[col].median(skipna=True)
                num_fill_values[col] = median if pd.notna(median) else 0

            na_artifacts = {
                'fill_value': fill_value,
                'cat_fill_values': cat_fill_values,
                'num_fill_values': num_fill_values,
            }
        else:
            na_artifacts = self.data_checkpoint.get("na_artifacts", {})

            fill_value = na_artifacts.get('fill_value', {})
            cat_fill_values = na_artifacts.get('cat_fill_values', {})
            num_fill_values = na_artifacts.get('num_fill_values', {})

            num_cols = [col for col in num_cols if col not in fill_value]
            cat_cols = [col for col in cat_cols if col not in fill_value]



        for col_name, value in fill_value.items():
            if col_name in X.columns:
                X[col_name] = X[col_name].fillna(value)
        

        for col in cat_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(cat_fill_values[col])
        
        for col in num_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(num_fill_values[col])
        
        return X, na_artifacts

    def _preprocessing(self, X, num_cols, cat_cols, preprocess_cfg, is_transform):
        num_cols = [col for col in num_cols if col in X.columns]
        cat_cols = [col for col in cat_cols if col in X.columns]

        scale_num = bool(preprocess_cfg.get("scale_num", False))
        encode_cat = preprocess_cfg.get("encode_cat", "onehot")
        onehot_max_cardinality = int(
            preprocess_cfg.get(
                "onehot_max_cardinality",
                getattr(self.cfg, "onehot_max_cardinality", 100),
            )
        )
        high_cardinality_strategy = preprocess_cfg.get(
            "high_cardinality_strategy",
            getattr(self.cfg, "high_cardinality_strategy", "frequency"),
        )

        if not is_transform:
            scaler, encoder = None, None
            low_card_cols = list(cat_cols)
            high_card_artifacts = {
                "strategy": high_cardinality_strategy,
                "columns": [],
                "frequency_maps": {},
            }

            if scale_num and num_cols:
                scaler = StandardScaler()
                scaler.fit(X[num_cols])

            if encode_cat == "onehot" and cat_cols:
                high_card_cols = [
                    col for col in cat_cols
                    if X[col].nunique(dropna=False) > onehot_max_cardinality
                ]
                low_card_cols = [col for col in cat_cols if col not in high_card_cols]

                if low_card_cols:
                    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    encoder.fit(X[low_card_cols])

                if high_card_cols:
                    high_card_artifacts["columns"] = list(high_card_cols)

                    if high_cardinality_strategy == "frequency":
                        high_card_artifacts["frequency_maps"] = {
                            col: X[col].value_counts(dropna=False, normalize=True).to_dict()
                            for col in high_card_cols
                        }
                    elif high_cardinality_strategy != "drop":
                        raise ValueError(
                            f"Unsupported high_cardinality_strategy: {high_cardinality_strategy}"
                        )
            elif encode_cat == "ordinal" and cat_cols:
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                    dtype=np.float32,
                )
                encoder.fit(X[cat_cols])

            pp_artifacts = {
                "scaler": scaler,
                "encoder": encoder,
                "encode_cat": encode_cat,
                "scale_num": scale_num,
                "onehot_low_card_cols": low_card_cols,
                "high_card_artifacts": high_card_artifacts,
            }
        else:
            pp_artifacts = self.data_checkpoint.get("pp_artifacts", {})
            scaler = pp_artifacts.get("scaler")
            encoder = pp_artifacts.get("encoder")
            low_card_cols = [
                col for col in pp_artifacts.get("onehot_low_card_cols", [])
                if col in X.columns
            ]
            high_card_artifacts = pp_artifacts.get("high_card_artifacts", {})
            high_card_cols = [
                col for col in high_card_artifacts.get("columns", [])
                if col in X.columns
            ]

        if scaler is not None:
            X_num = pd.DataFrame(
                scaler.transform(X[num_cols]),
                columns=num_cols,
                index=X.index,
            )
        else:
            X_num = X.loc[:, num_cols]

        out_num_cols = list(X_num.columns)

        if encode_cat == "onehot":
            if encoder is not None and low_card_cols:
                cat_feature_names = list(encoder.get_feature_names_out(low_card_cols))
                X_cat = pd.DataFrame(
                    encoder.transform(X[low_card_cols]),
                    columns=cat_feature_names,
                    index=X.index,
                )
            else:
                cat_feature_names = []
                X_cat = pd.DataFrame(index=X.index)

            high_card_feature_names = []
            X_high_card = pd.DataFrame(index=X.index)
            if high_card_cols:
                strategy = high_card_artifacts.get("strategy", high_cardinality_strategy)
                if strategy == "frequency":
                    freq_columns = {}
                    frequency_maps = high_card_artifacts.get("frequency_maps", {})
                    for col in high_card_cols:
                        freq_columns[f"{col}__freq"] = (
                            X[col].map(frequency_maps.get(col, {})).fillna(0.0).astype(np.float32)
                        )

                    X_high_card = pd.DataFrame(freq_columns, index=X.index)
                    high_card_feature_names = list(X_high_card.columns)
                elif strategy != "drop":
                    raise ValueError(f"Unsupported high_cardinality_strategy: {strategy}")

            out_cat_cols = []
            if not X_cat.empty:
                X_cat = X_cat.astype(np.float32)
                out_num_cols.extend(list(X_cat.columns))
            if not X_high_card.empty:
                out_num_cols.extend(high_card_feature_names)
        elif encode_cat == "ordinal":
            if encoder is not None and cat_cols:
                cat_feature_names = list(cat_cols)
                X_cat = pd.DataFrame(
                    encoder.transform(X[cat_cols]),
                    columns=cat_feature_names,
                    index=X.index,
                ).astype(np.float32)
            else:
                cat_feature_names = []
                X_cat = pd.DataFrame(index=X.index)

            out_cat_cols = []
            if not X_cat.empty:
                out_num_cols.extend(list(X_cat.columns))
            X_high_card = pd.DataFrame(index=X.index)
            high_card_feature_names = []
        elif encode_cat in {"none", None}:
            X_cat = X.loc[:, cat_cols]
            cat_feature_names = list(X_cat.columns)
            out_cat_cols = list(X_cat.columns)
            X_high_card = pd.DataFrame(index=X.index)
            high_card_feature_names = []
        else:
            raise ValueError(f"Unsupported encode_cat mode: {encode_cat}")

        X_out = pd.concat([X_num, X_cat, X_high_card], axis=1)
        pp_artifacts["cat_feature_names"] = cat_feature_names
        pp_artifacts["high_card_feature_names"] = high_card_feature_names
        return X_out, out_num_cols, out_cat_cols, pp_artifacts

    def _feature_engineering_before_na(self, X, num_cols, cat_cols):
        title_mapping = getattr(self.cfg, "title_mapping", {})

        # X["Initial"] = X["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
        # X["Initial"] = X["Initial"].replace(title_mapping)
        # X["Initial"] = X["Initial"].fillna("Other")
        # cat_cols = self._append_if_missing(cat_cols, "Initial")

        return X, num_cols, cat_cols

    def _feature_engineering_after_na(self, X, num_cols, cat_cols):
        # fare_bins = getattr(self.cfg, "fare_bins", None)
        # fare_labels = getattr(self.cfg, "fare_labels", None)

        # X["Family_Size"] = (X["Parch"] + X["SibSp"] + 1).astype(int)
        # X["Alone"] = (X["Family_Size"] == 1).astype(int).astype(str)

        # X["Fare_cat"] = pd.cut(
        #     X["Fare"],
        #     bins=fare_bins,
        #     labels=fare_labels,
        #     include_lowest=True,
        # ).astype(str)

        # num_cols = self._append_if_missing(num_cols, "Family_Size")
        # cat_cols = self._append_if_missing(cat_cols, "Alone")
        # cat_cols = self._append_if_missing(cat_cols, "Fare_cat")

        return X, num_cols, cat_cols

    def _drop(self, X, num_cols, cat_cols):
        checkpoint_drop_columns = self.data_checkpoint.get("drop_columns")
        checkpoint_constant_columns = self.data_checkpoint.get("constant_columns")

        if checkpoint_drop_columns is not None:
            drop_columns = [col for col in checkpoint_drop_columns if col in X.columns]
            constant_columns = [
                col for col in (checkpoint_constant_columns or [])
                if col in X.columns
            ]
        else:
            config_drop_columns = [col for col in self.cfg.drop_columns if col in X.columns]
            constant_columns = [
                col for col in X.columns
                if col not in config_drop_columns and X[col].nunique(dropna=False) == 1
            ]

            # for col in constant_columns:
            #     if col not in self._reported_constant_columns:
            #         self._reported_constant_columns.add(col)
            #         print(f'!!!const column: {col}')

            drop_columns = config_drop_columns + constant_columns

        if drop_columns:
            X = X.drop(columns=drop_columns)

        num_cols = [col for col in num_cols if col not in drop_columns]
        cat_cols = [col for col in cat_cols if col not in drop_columns]

        return X, drop_columns, constant_columns, num_cols, cat_cols

    # === ====== ===
    # Вспомагательные функции
    # === ====== ===
    def _infer_feature_types(self, X):
        force_categorical = set(
            self.data_checkpoint.get(
                "force_categorical_columns",
                getattr(self.cfg, "force_categorical_columns", []),
            )
        )

        cat_cols = []
        num_cols = []

        for col in X.columns:
            if col in force_categorical:
                cat_cols.append(col)
            elif (
                is_object_dtype(X[col])
                or isinstance(X[col].dtype, CategoricalDtype)
                or is_bool_dtype(X[col])
            ):
                cat_cols.append(col)
            elif is_numeric_dtype(X[col]):
                num_cols.append(col)
            else:
                cat_cols.append(col)

        return num_cols, cat_cols
        
    def _standardization_df(self, X, num_cols, cat_cols):
        num_cols = [col for col in num_cols if col in X.columns]
        cat_cols = [col for col in cat_cols if col in X.columns]

        all_cols = num_cols + cat_cols
        X = X.reindex(columns=all_cols)

        if num_cols:
            X = X.astype({col: np.float32 for col in num_cols})

        if cat_cols:
            for col in cat_cols:
                # CatBoost requires categorical values to be strings or integers.
                # Some object columns contain float-like values (for example 18.99),
                # so we normalize all categorical features to string categories.
                X[col] = X[col].astype("string").fillna("missing").astype("category")

        return X, num_cols, cat_cols

    def _resolve_preprocess_cfg(self):
        checkpoint_cfg = self.data_checkpoint.get("preprocess_cfg")
        if checkpoint_cfg:
            return checkpoint_cfg

        profile_map = getattr(self.cfg, "model_preprocess_profile", {}) or {}
        profile_name = profile_map.get(self.model_name)
        if profile_name is None:
            raise ValueError(f"No preprocessing profile for model '{self.model_name}'.")

        profiles = getattr(self.cfg, "preprocessing_profiles", {}) or {}
        profile_cfg = profiles.get(profile_name)
        if profile_cfg is None:
            raise ValueError(f"Profile '{profile_name}' not found in cfg.preprocessing_profiles.")

        preprocess_cfg = dict(profile_cfg)
        preprocess_cfg["profile_name"] = profile_name
        return preprocess_cfg

    @staticmethod
    def _append_if_missing(columns, column_name):
        if column_name not in columns:
            columns.append(column_name)
        return columns

    def save_data_checkpoint(self):
        path = self.cfg.path_to_save_data_checkpoint
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.data_checkpoint, f)

    def load_data_checkpoint(self):
        with open(self.cfg.path_to_load_data_checkpoint, "rb") as f:
            self.data_checkpoint = pickle.load(f)
