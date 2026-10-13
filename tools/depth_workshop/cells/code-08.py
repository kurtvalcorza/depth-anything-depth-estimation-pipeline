reference = [2.0, 4.0]
prediction = [2.5, 4.0]
errors = [abs(p - r) / r for p, r in zip(prediction, reference)]
passes = [max(p / r, r / p) < 1.25 for p, r in zip(prediction, reference)]
print({'AbsRel': sum(errors) / len(errors), 'delta1': sum(passes) / len(passes)})
print('Inverse depths of 2 m and 4 m:', [1 / r for r in reference])