from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


FEATURES = ["home_elo", "away_elo", "elo_diff", "neutral", "home_form", "away_form"]
REQUIRED_DATA_FILES = ["results.csv", "shootouts.csv", "eloratings.csv"]
CACHE_VERSION = 2


class SetupError(RuntimeError):
    """Raised when the ML pipeline cannot be trained from the local project."""


@dataclass
class Prediction:
    home: str
    away: str
    model: str
    predicted_result: str
    winner: str | None
    probabilities: dict[str, float]
    features: dict[str, float]


class WorldCupModelPipeline:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.cache_dir = self.project_root / "backend" / "cache"
        self.trained = False
        self.models: dict[str, Any] = {}
        self.encoder = None
        self.eloratings = None
        self.all_team_results = None
        self.team_stats: dict[str, tuple[float, float]] = {}
        self.avg_elo = 1500.0
        self.avg_form = 1.0
        self.metrics: dict[str, Any] = {}

    def dependency_status(self) -> dict[str, bool]:
        return {
            "pandas": self._has_module("pandas"),
            "numpy": self._has_module("numpy"),
            "sklearn": self._has_module("sklearn"),
            "xgboost": self._has_module("xgboost"),
            "joblib": self._has_module("joblib"),
        }

    def data_status(self) -> dict[str, bool]:
        return {name: (self.data_dir / name).exists() for name in REQUIRED_DATA_FILES}

    def status(self) -> dict[str, Any]:
        deps = self.dependency_status()
        data = self.data_status()
        ready = all(data.values()) and deps["pandas"] and deps["numpy"] and deps["sklearn"]
        return {
            "ready": ready,
            "trained": self.trained,
            "dependencies": deps,
            "data": data,
            "available_models": list(self.models.keys()),
            "cache": self.cache_status(),
            "metrics": self.metrics,
        }

    def train_if_needed(self, model_name: str = "rf") -> None:
        if model_name in self.models and self.trained:
            return
        if self.load_cache(model_name):
            return
        self.train(model_name)

    def train(self, model_name: str = "rf") -> None:
        if model_name not in {"rf", "xgb"}:
            model_name = "rf"
        missing_data = [name for name, exists in self.data_status().items() if not exists]
        if missing_data:
            raise SetupError(f"Missing data files in data/: {', '.join(missing_data)}")

        deps = self.dependency_status()
        missing_deps = [name for name in ("pandas", "numpy", "sklearn") if not deps[name]]
        if model_name == "xgb" and not deps["xgboost"]:
            missing_deps.append("xgboost")
        if missing_deps:
            raise SetupError(f"Missing Python packages: {', '.join(missing_deps)}")

        import pandas as pd

        RandomForestClassifier = import_module("sklearn.ensemble").RandomForestClassifier
        accuracy_score = import_module("sklearn.metrics").accuracy_score
        model_selection = import_module("sklearn.model_selection")
        train_test_split = model_selection.train_test_split
        LabelEncoder = import_module("sklearn.preprocessing").LabelEncoder

        results = pd.read_csv(self.data_dir / "results.csv")
        shootouts = pd.read_csv(self.data_dir / "shootouts.csv")
        eloratings = pd.read_csv(self.data_dir / "eloratings.csv")

        results["date"] = pd.to_datetime(results["date"])
        shootouts["date"] = pd.to_datetime(shootouts["date"])
        eloratings["date"] = pd.to_datetime(eloratings["date"], format="mixed")
        eloratings["team"] = eloratings["team"].astype(str).str.replace("\xa0", " ", regex=False)

        results = results.dropna(subset=["home_score", "away_score"]).copy()
        results["home_score"] = results["home_score"].astype(int)
        results["away_score"] = results["away_score"].astype(int)
        results = results.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

        if "first_shooter" in shootouts.columns:
            shootouts = shootouts.drop(columns=["first_shooter"])

        results = results.merge(
            shootouts[["date", "home_team", "away_team", "winner"]],
            on=["date", "home_team", "away_team"],
            how="left",
        )
        results["result"] = results.apply(self._result_from_row, axis=1)

        results = results.sort_values("date")
        eloratings = eloratings.sort_values("date")

        results = pd.merge_asof(
            results,
            eloratings[["date", "team", "rating"]].rename(columns={"rating": "home_elo"}),
            left_on="date",
            right_on="date",
            left_by="home_team",
            right_by="team",
            direction="backward",
        ).drop(columns=["team"])

        results = pd.merge_asof(
            results,
            eloratings[["date", "team", "rating"]].rename(columns={"rating": "away_elo"}),
            left_on="date",
            right_on="date",
            left_by="away_team",
            right_by="team",
            direction="backward",
        ).drop(columns=["team"])

        results["elo_diff"] = results["home_elo"] - results["away_elo"]
        results["neutral"] = results["neutral"].astype(int)

        all_team_results = self._build_form_table(results)
        wc_all = results[results["tournament"] == "FIFA World Cup"].dropna(
            subset=["home_elo", "away_elo"]
        ).copy()

        wc_all = wc_all.merge(
            all_team_results[["date", "team", "form_5"]],
            left_on=["date", "home_team"],
            right_on=["date", "team"],
            how="left",
        ).drop(columns="team").rename(columns={"form_5": "home_form"})
        wc_all = wc_all.merge(
            all_team_results[["date", "team", "form_5"]],
            left_on=["date", "away_team"],
            right_on=["date", "team"],
            how="left",
        ).drop(columns="team").rename(columns={"form_5": "away_form"})
        wc_all["home_form"] = wc_all["home_form"].fillna(1.0)
        wc_all["away_form"] = wc_all["away_form"].fillna(1.0)

        x = wc_all[FEATURES].copy()
        y = wc_all["result"].copy()
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y_encoded, test_size=0.2, random_state=42
        )

        metrics: dict[str, Any] = {
            "training_rows": int(len(wc_all)),
            "features": FEATURES,
            "target_mapping": dict(zip(encoder.classes_, encoder.transform(encoder.classes_).tolist())),
        }

        if model_name == "rf":
            model = RandomForestClassifier(random_state=42)
            model.fit(x_train, y_train)
            self.models["rf"] = model
            metrics["rf_accuracy"] = float(accuracy_score(y_test, model.predict(x_test)))
            metrics["rf_cv_mean"] = None
        else:
            XGBClassifier = import_module("xgboost").XGBClassifier

            model = XGBClassifier(random_state=42, eval_metric="mlogloss")
            model.fit(x_train, y_train)
            self.models["xgb"] = model
            metrics["xgb_accuracy"] = float(accuracy_score(y_test, model.predict(x_test)))
            metrics["xgb_cv_mean"] = None

        self.encoder = encoder
        self.eloratings = eloratings
        self.all_team_results = all_team_results
        self.avg_elo = float(eloratings["rating"].mean())
        self.avg_form = float(all_team_results["form_5"].dropna().mean())
        self.team_stats = self._build_team_stats(eloratings, all_team_results)
        self.metrics = metrics
        self.trained = True
        self.save_cache(model_name)

    def cache_status(self) -> dict[str, Any]:
        return {model: self.model_cache_status(model) for model in ("rf", "xgb")}

    def model_cache_status(self, model_name: str) -> dict[str, Any]:
        path = self.cache_path(model_name)
        valid = False
        if path.exists():
            try:
                cached = self._load_cache_payload(model_name)
                valid = (
                    cached.get("cache_version") == CACHE_VERSION
                    and cached.get("model_name") == model_name
                    and cached.get("data_fingerprint") == self.data_fingerprint()
                )
            except Exception:
                valid = False
        return {"path": str(path), "exists": path.exists(), "valid": valid}

    def load_cache(self, model_name: str = "rf") -> bool:
        if not self.cache_path(model_name).exists() or not self._has_module("joblib"):
            return False
        try:
            cached = self._load_cache_payload(model_name)
        except Exception:
            return False

        if cached.get("cache_version") != CACHE_VERSION:
            return False
        if cached.get("model_name") != model_name:
            return False
        if cached.get("data_fingerprint") != self.data_fingerprint():
            return False

        self.models[model_name] = cached["model"]
        self.encoder = cached["encoder"]
        self.eloratings = cached["eloratings"]
        self.all_team_results = cached["all_team_results"]
        self.team_stats = cached.get("team_stats", {})
        self.avg_elo = cached["avg_elo"]
        self.avg_form = cached["avg_form"]
        self.metrics = {**cached["metrics"], "loaded_from_cache": True}
        self.trained = True
        return True

    def save_cache(self, model_name: str = "rf") -> None:
        if not self._has_module("joblib"):
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        joblib = import_module("joblib")
        joblib.dump(
            {
                "cache_version": CACHE_VERSION,
                "model_name": model_name,
                "data_fingerprint": self.data_fingerprint(),
                "model": self.models[model_name],
                "encoder": self.encoder,
                "eloratings": self.eloratings,
                "all_team_results": self.all_team_results,
                "team_stats": self.team_stats,
                "avg_elo": self.avg_elo,
                "avg_form": self.avg_form,
                "metrics": {**self.metrics, "loaded_from_cache": False},
            },
            self.cache_path(model_name),
        )

    def data_fingerprint(self) -> dict[str, dict[str, float | int]]:
        fingerprint = {}
        for name in REQUIRED_DATA_FILES:
            path = self.data_dir / name
            if not path.exists():
                continue
            stat = path.stat()
            fingerprint[name] = {"size": stat.st_size, "mtime": stat.st_mtime}
        return fingerprint

    def cache_path(self, model_name: str) -> Path:
        return self.cache_dir / f"worldcup_model_cache_{model_name}.joblib"

    def _load_cache_payload(self, model_name: str) -> dict[str, Any]:
        joblib = import_module("joblib")
        return joblib.load(self.cache_path(model_name))

    def predict_match(self, home: str, away: str, model_name: str = "rf") -> Prediction:
        self.train_if_needed(model_name)
        if model_name not in self.models:
            model_name = "rf"
        if self.encoder is None:
            raise SetupError("Model encoder is not available. Train the pipeline first.")

        import pandas as pd

        home_elo, home_form = self.get_stats(home)
        away_elo, away_form = self.get_stats(away)
        features = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "neutral": 1.0,
            "home_form": home_form,
            "away_form": away_form,
        }
        input_data = pd.DataFrame([features], columns=FEATURES)
        model = self.models[model_name]
        raw_probs = model.predict_proba(input_data)[0]
        predicted_index = int(model.predict(input_data)[0])
        predicted_result = str(self.encoder.inverse_transform([predicted_index])[0])
        probabilities = {
            str(label): float(raw_probs[index])
            for index, label in enumerate(self.encoder.classes_)
        }
        winner = None
        if predicted_result == "home_win":
            winner = home
        elif predicted_result == "away_win":
            winner = away

        return Prediction(
            home=home,
            away=away,
            model=model_name,
            predicted_result=predicted_result,
            winner=winner,
            probabilities=probabilities,
            features={key: float(value) for key, value in features.items()},
        )

    def get_stats(self, team: str) -> tuple[float, float]:
        if self.eloratings is None or self.all_team_results is None:
            raise SetupError("Team statistics are not available. Train the pipeline first.")

        elo_team = normalize_team_name(team)
        return self.team_stats.get(elo_team, (self.avg_elo, self.avg_form))

    @staticmethod
    def _has_module(name: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(name) is not None

    @staticmethod
    def _result_from_row(row: Any) -> str:
        if row["home_score"] > row["away_score"]:
            return "home_win"
        if row["away_score"] > row["home_score"]:
            return "away_win"
        if row.get("winner") == row["home_team"]:
            return "home_win"
        if row.get("winner") == row["away_team"]:
            return "away_win"
        return "draw"

    @staticmethod
    def _build_form_table(results: Any) -> Any:
        def points(result: str, team_type: str) -> int:
            if result == "draw":
                return 1
            if team_type == "home" and result == "home_win":
                return 3
            if team_type == "away" and result == "away_win":
                return 3
            return 0

        import pandas as pd

        home_results = results[["date", "home_team", "result"]].rename(columns={"home_team": "team"})
        home_results["points"] = home_results["result"].map(lambda result: points(result, "home"))
        away_results = results[["date", "away_team", "result"]].rename(columns={"away_team": "team"})
        away_results["points"] = away_results["result"].map(lambda result: points(result, "away"))
        all_team_results = pd.concat([home_results, away_results]).sort_values(["team", "date"])
        all_team_results["form_5"] = all_team_results.groupby("team")["points"].transform(
            lambda values: values.shift().rolling(5).mean()
        )
        return all_team_results

    @staticmethod
    def _build_team_stats(eloratings: Any, all_team_results: Any) -> dict[str, tuple[float, float]]:
        stats: dict[str, tuple[float, float]] = {}

        elo_frame = eloratings.copy()
        elo_frame["normalized_team"] = elo_frame["team"].map(normalize_team_name)
        latest_elos = elo_frame.sort_values("date").drop_duplicates("normalized_team", keep="last")

        form_frame = all_team_results.dropna(subset=["form_5"]).copy()
        form_frame["normalized_team"] = form_frame["team"].map(normalize_team_name)
        latest_forms = form_frame.sort_values("date").drop_duplicates("normalized_team", keep="last")
        form_lookup = dict(zip(latest_forms["normalized_team"], latest_forms["form_5"]))

        for _, row in latest_elos.iterrows():
            team = row["normalized_team"]
            stats[team] = (float(row["rating"]), float(form_lookup.get(team, 1.0)))

        return stats


def normalize_team_name(name: str) -> str:
    aliases = {
        "curacao": "curacao",
        "cote d'ivoire": "cote d'ivoire",
        "cote d ivoire": "cote d'ivoire",
        "ivory coast": "cote d'ivoire",
        "usa": "united states",
        "united states": "united states",
        "south korea": "korea republic",
        "korea republic": "korea republic",
    }
    cleaned = (
        str(name)
        .lower()
        .replace("côte", "cote")
        .replace("curaçao", "curacao")
        .replace(".", "")
        .strip()
    )
    return aliases.get(cleaned, cleaned)
