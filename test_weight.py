from src.weight_estimator import WeightEstimator

tracks = [
    [10, 10, 60, 60, 1],
    [20, 20, 80, 90, 2]
]

estimator = WeightEstimator(scale_factor=0.001)

weights = estimator.estimate(tracks)
avg = estimator.average_weight(weights)

print("Weights:", weights)
print("Average:", avg)
