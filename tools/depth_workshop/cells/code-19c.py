RUN_OWN_FACTOR = False  # @param {type:"boolean"}
OWN_BRIGHTNESS_FACTOR = 0.8  # @param {type:"number"}
if RUN_OWN_FACTOR:
    if not 0.2 <= float(OWN_BRIGHTNESS_FACTOR) <= 2.0:
        raise ValueError('Choose OWN_BRIGHTNESS_FACTOR between 0.2 and 2.0, then rerun this cell.')
    run_stage('own_activity', '--factor', str(float(OWN_BRIGHTNESS_FACTOR)))
    own = json.loads((ROOT / 'own_activity_output.json').read_text())
    comparison = own['comparison']
    kept = [r for r in comparison['rows'] if r.get('informative', True)]
    mean_delta = sum(r['abs_rel_delta'] for r in kept) / len(kept)
    print({'factor': own['factor'], 'improved': comparison['improved'],
           'unchanged': comparison['unchanged'], 'worsened': comparison['worsened'],
           'mean_paired_abs_rel_delta': round(mean_delta, 5)})
    from IPython.display import Image as ShowImage, display
    display(ShowImage(filename=str(Path(own['root']) / 'outputs/own_factor.png')))
else:
    print('Optional own-factor activity disabled.')
