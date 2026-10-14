run_stage('depth')
from IPython.display import Markdown, display
history = json.loads((ROOT / 'outputs/training_history.json').read_text())
observed = {o['epoch']: o for o in history['epoch_observations']}
table = ['| Epoch | Training loss | Validation aligned AbsRel | Delta-1 | Optimizer steps so far | '
         'Max weight change before selection |', '|---:|---:|---:|---:|---:|---:|']
for entry in history['history']:
    loss = 'baseline' if entry['train_loss'] is None else f"{entry['train_loss']:.6f}"
    seen = observed.get(entry['epoch'])
    steps = 0 if seen is None else seen['actual_steps']
    change = '0 (starting weights)' if seen is None else f"{seen['max_weight_delta_before_selection']:.3e}"
    table.append(f"| {entry['epoch']} | {loss} | {entry['val']['abs_rel']:.6f} | {entry['val']['delta1']:.6f} "
                 f"| {steps} | {change} |")
display(Markdown('\n'.join(table)))
print({'actual_optimizer_steps': history['actual_optimizer_steps'],
       'trainable_parameters': history['trainable_parameters'],
       'selected_max_weight_delta_after_restoring_selected_epoch': history['selected_max_weight_delta']})
print('Full tensor hashes and training evidence are saved in training_history.json.')
show_json('outputs/selection.json')
show_metric_tables(['depth_frozen.json', 'depth_selected_live.json'], breakdown=True)
show_previews('depth*.png')