baselines = json.loads((ROOT / 'outputs/baselines.json').read_text())
print('Training-only constant (metres):', baselines['training_constant_metres'])
show_metric_tables(['baselines.json'])