from __future__ import annotations

from itertools import combinations
from typing import Any


GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Cote d'Ivoire", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def simulate_tournament(
    pipeline: Any,
    model_name: str = "rf",
    include_group_matches: bool = False,
) -> dict[str, Any]:
    group_tables: dict[str, list[dict[str, Any]]] = {}
    group_matches: dict[str, list[dict[str, Any]]] = {}

    for group, teams in GROUPS_2026.items():
        table = {
            team: {"team": team, "group": group, "played": 0, "points": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in teams
        }
        matches = []

        for home, away in combinations(teams, 2):
            prediction = pipeline.predict_match(home, away, model_name)
            home_goals, away_goals = scoreline_from_prediction(prediction)
            apply_group_result(table[home], table[away], home_goals, away_goals)
            matches.append(
                {
                    "home": home,
                    "away": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "prediction": prediction.__dict__,
                }
            )

        sorted_table = sorted(
            table.values(),
            key=lambda row: (
                row["points"],
                row["gd"],
                row["gf"],
                pipeline.get_stats(row["team"])[0],
            ),
            reverse=True,
        )
        group_tables[group] = sorted_table
        group_matches[group] = matches

    thirds = sorted(
        [table[2] for table in group_tables.values()],
        key=lambda row: (row["points"], row["gd"], row["gf"], pipeline.get_stats(row["team"])[0]),
        reverse=True,
    )
    bracket = build_bracket(pipeline, group_tables, thirds, model_name)
    champion = bracket[-1]["matches"][0]["winner"]

    result = {
        "source": "backend",
        "model": model_name if model_name in pipeline.models else "rf",
        "status": pipeline.status(),
        "groups": group_tables,
        "thirds": thirds,
        "bracket": bracket,
        "champion": champion,
    }
    if include_group_matches:
        result["group_matches"] = group_matches
    return result


def compare_models(pipeline: Any) -> dict[str, Any]:
    rf = simulate_tournament(pipeline, "rf")
    xgb = simulate_tournament(pipeline, "xgb")
    return {
        "models": {
            "rf": summarize_simulation(rf),
            "xgb": summarize_simulation(xgb),
        },
        "group_disagreements": compare_group_tables(rf["groups"], xgb["groups"]),
        "bracket_disagreements": compare_brackets(rf["bracket"], xgb["bracket"]),
        "status": pipeline.status(),
    }


def summarize_simulation(simulation: dict[str, Any]) -> dict[str, Any]:
    final = simulation["bracket"][-1]["matches"][0]
    return {
        "champion": simulation["champion"],
        "final": {"home": final["home"], "away": final["away"], "winner": final["winner"]},
        "model": simulation["model"],
    }


def compare_group_tables(
    rf_groups: dict[str, list[dict[str, Any]]],
    xgb_groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    disagreements = []
    for group in rf_groups:
        rf_table = rf_groups[group]
        xgb_table = xgb_groups[group]
        rf_top_three = [row["team"] for row in rf_table[:3]]
        xgb_top_three = [row["team"] for row in xgb_table[:3]]
        if rf_top_three != xgb_top_three:
            disagreements.append(
                {
                    "group": group,
                    "rf_top_three": rf_top_three,
                    "xgb_top_three": xgb_top_three,
                    "rf_winner": rf_top_three[0],
                    "xgb_winner": xgb_top_three[0],
                }
            )
    return disagreements


def compare_brackets(rf_bracket: list[dict[str, Any]], xgb_bracket: list[dict[str, Any]]) -> list[dict[str, Any]]:
    disagreements = []
    for round_index, rf_round in enumerate(rf_bracket):
        xgb_round = xgb_bracket[round_index]
        for match_index, rf_match in enumerate(rf_round["matches"]):
            if match_index >= len(xgb_round["matches"]):
                continue
            xgb_match = xgb_round["matches"][match_index]
            if rf_match["winner"] != xgb_match["winner"]:
                disagreements.append(
                    {
                        "round": rf_round["name"],
                        "match": match_index + 1,
                        "rf": {
                            "home": rf_match["home"],
                            "away": rf_match["away"],
                            "winner": rf_match["winner"],
                        },
                        "xgb": {
                            "home": xgb_match["home"],
                            "away": xgb_match["away"],
                            "winner": xgb_match["winner"],
                        },
                    }
                )
    return disagreements


def build_bracket(pipeline: Any, group_tables: dict[str, list[dict[str, Any]]], thirds: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    third_teams = [row["team"] for row in thirds[:8]]

    def w(group: str) -> str:
        return group_tables[group][0]["team"]

    def ru(group: str) -> str:
        return group_tables[group][1]["team"]

    round_of_32 = [
        [ru("A"), ru("B")], [w("F"), ru("C")], [w("E"), third_teams[0]], [w("I"), third_teams[1]],
        [w("C"), ru("F")], [ru("E"), ru("I")], [w("A"), third_teams[2]], [w("L"), third_teams[3]],
        [ru("K"), ru("L")], [w("H"), ru("J")], [w("D"), third_teams[4]], [w("G"), third_teams[5]],
        [w("J"), ru("H")], [ru("D"), ru("G")], [w("B"), third_teams[6]], [w("K"), third_teams[7]],
    ]

    rounds = [{"name": "Round of 32", "matches": knockout_matches(pipeline, round_of_32, model_name)}]
    previous = [match["winner"] for match in rounds[0]["matches"]]

    for name in ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]:
        pairs = [[previous[index], previous[index + 1]] for index in range(0, len(previous), 2)]
        matches = knockout_matches(pipeline, pairs, model_name)
        rounds.append({"name": name, "matches": matches})
        previous = [match["winner"] for match in matches]

    return rounds


def knockout_matches(pipeline: Any, pairs: list[list[str]], model_name: str) -> list[dict[str, Any]]:
    matches = []
    for home, away in pairs:
        prediction = pipeline.predict_match(home, away, model_name)
        winner = prediction.winner
        if winner is None:
            home_prob = prediction.probabilities.get("home_win", 0.0)
            away_prob = prediction.probabilities.get("away_win", 0.0)
            if home_prob == away_prob:
                winner = home if prediction.features["home_elo"] >= prediction.features["away_elo"] else away
            else:
                winner = home if home_prob > away_prob else away
        matches.append({"home": home, "away": away, "winner": winner, "prediction": prediction.__dict__})
    return matches


def apply_group_result(home_row: dict[str, Any], away_row: dict[str, Any], home_goals: int, away_goals: int) -> None:
    home_row["played"] += 1
    away_row["played"] += 1
    home_row["gf"] += home_goals
    home_row["ga"] += away_goals
    away_row["gf"] += away_goals
    away_row["ga"] += home_goals
    home_row["gd"] = home_row["gf"] - home_row["ga"]
    away_row["gd"] = away_row["gf"] - away_row["ga"]

    if home_goals > away_goals:
        home_row["points"] += 3
    elif away_goals > home_goals:
        away_row["points"] += 3
    else:
        home_row["points"] += 1
        away_row["points"] += 1


def scoreline_from_prediction(prediction: Any) -> tuple[int, int]:
    home_prob = prediction.probabilities.get("home_win", 0.0)
    away_prob = prediction.probabilities.get("away_win", 0.0)
    draw_prob = prediction.probabilities.get("draw", 0.0)

    if prediction.predicted_result == "draw":
        return (1, 1) if draw_prob < 0.55 else (0, 0)

    confidence = max(home_prob, away_prob)
    margin = 1
    if confidence > 0.62:
        margin = 2
    if confidence > 0.78:
        margin = 3

    if prediction.predicted_result == "home_win":
        return margin, 0 if away_prob < 0.28 else 1
    return 0 if home_prob < 0.28 else 1, margin
