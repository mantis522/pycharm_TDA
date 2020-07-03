from ax import optimize
best_parameters, best_values, experiment, model = optimize(
    parameters=[
        {
        "name": "x1",
        "type": "range",
        "bounds": [-10.0, 10.0],
          },
          {
            "name": "x2",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
        ],
        # Booth function
        evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
        minimize=True,
    )

print(best_parameters)