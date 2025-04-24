import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from utils_and_constants import PROCESSED_DATASET, get_hp_tuning_results, load_data


def main():
    X, y = load_data(PROCESSED_DATASET)
    X_train, _, y_train, _ = train_test_split(X, y, random_state=1993)

    model = RandomForestClassifier()
    # Read the config file to define the hyperparameter search space
    param_grid = json.load(open("hp_config.json", "r"))

    # Perform Grid Search Cross Validation on training data
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    print("====================Best Hyperparameters==================")
    print(json.dumps(best_params, indent=2))
    print("==========================================================")

    with open("rfc_best_params.json", "w") as outfile:
        json.dump(best_params, outfile)

    markdown_table = get_hp_tuning_results(grid_search)
    with open("hp_tuning_results.md", "w") as markdown_file:
        markdown_file.write(markdown_table)


if __name__ == "__main__":
    main()
