run_stage('reload')
parity = json.loads((ROOT / 'outputs/reload_parity.json').read_text())
print({'passed': parity['passed'], 'images_checked': len(parity['checks']),
       'max_absolute_prediction_error': max(r['max_absolute_error'] for r in parity['checks']),
       'metric_drift': parity['metric_drift'], 'reload_pid': parity['reload_pid']})