#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
# def solve(data):
    ### BEGIN SOLUTION
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2,
                                                        random_state=42)
    ### END SOLUTION
    # return x_train, y_train, x_test, y_test
# x_train, y_train, x_test, y_test = solve(data)


    return x_train, y_train, x_test, y_test

PY
