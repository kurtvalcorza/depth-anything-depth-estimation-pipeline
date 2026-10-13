run_stage('report', echo=False)  # summary.md is shown once below, rendered
import shutil
from IPython.display import Markdown, display
display(Markdown((ROOT / 'outputs/summary.md').read_text()))
show_previews('figures/*.png')
receipts = json.loads((ROOT / 'stage_receipts.json').read_text())['stages']
resources = {stage: {'seconds': round(r.get('seconds', 0.0), 1),
                     'peak_gpu_mib': None if r.get('peak_gpu_bytes') is None else round(r['peak_gpu_bytes'] / 2**20, 1)}
             for stage, r in receipts.items()}
run_folder_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
disk = shutil.disk_usage(ROOT)
print('This run, per stage (wall seconds, peak GPU MiB):', json.dumps(resources, indent=2))
print(f'Run folder on disk: {run_folder_bytes / 2**30:.2f} GiB (environment, models, data and outputs); '
      f'free disk: {disk.free / 2**30:.1f} GiB')
print('Results ZIP:', ROOT / 'outputs/depth_results.zip')
print('To download it in Colab: open the Files pane (folder icon, left), browse to that path, right-click > Download.')
session = {'bootstrap_seconds': BOOTSTRAP_SECONDS, 'elapsed_seconds': time.perf_counter() - SESSION_START,
           'stage_resources': resources, 'run_folder_bytes': run_folder_bytes, 'disk_free_bytes': disk.free,
           'source_files': CARRIED_HASHES, 'qualification': 'Candidate until exact hosted and BYOD evidence reviewed'}
(ROOT / 'session_summary.json').write_text(json.dumps(session, indent=2))
print('Session summary (not inside the ZIP):', ROOT / 'session_summary.json')
print(session)