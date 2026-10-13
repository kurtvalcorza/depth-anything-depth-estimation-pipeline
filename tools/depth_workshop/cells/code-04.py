# @title Infrastructure: build the isolated Python 3.12.12 environment and define helpers
UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60'
for attempt in range(3):
    try:
        with urllib.request.urlopen(UV_URL, timeout=90) as response:
            wheel = response.read(20081405)
        break
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256:
    raise RuntimeError('uv wheel size/hash mismatch')
import io
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
ENV.pop('HF_TOKEN', None)
ENV.pop('HUGGING_FACE_HUB_TOKEN', None)
print('Creating the isolated Python 3.12.12 environment and installing the hash-locked packages;'
      ' this can take a few minutes on a fresh runtime.', flush=True)
subprocess.run([str(UV), 'venv', '--managed-python', '--python', '3.12.12', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:',
                '--index-url', 'https://pypi.org/simple', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
versions = subprocess.run([str(PYTHON), '-c', 'import sys, importlib.metadata as m, torch, transformers; '
                           'print("Python", sys.version.split()[0]); '
                           'print("torch", torch.__version__, "built for CUDA", torch.version.cuda); '
                           'print("transformers", transformers.__version__); '
                           'print(*(f"{p} {m.version(p)}" for p in ("torchvision", "huggingface-hub", '
                           '"safetensors", "numpy", "pillow")), sep="\\n"); '
                           'assert torch.cuda.is_available(), "CUDA is not available"; '
                           'print("device", torch.cuda.get_device_name(0))'],
                          env=ENV, capture_output=True, text=True)
print(versions.stdout.strip())
if versions.returncode:
    print(versions.stderr)
    raise RuntimeError('The isolated environment failed its version/CUDA check. Read the output above.')
print('Dependencies installed and verified in the isolated environment.', flush=True)
BOOTSTRAP_SECONDS = time.perf_counter() - SESSION_START

def run_stage(stage, *options, root=None, echo=True):
    # Streams the worker's output once (echo=False keeps it in the log only). Returns True when the stage
    # completed, False when the worker refused the learner's input (exit 65; it printed one 'Refused:' line).
    # Any other failure raises.
    selected_root = ROOT if root is None else root
    log = selected_root / (stage + '.log')
    print('Running', stage, '— one model per process; log:', log, flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'workshop.py'), '--root', str(selected_root), '--stage', stage, *options],
                                  env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in result.stdout:
            output.write(line)
            output.flush()
            if echo and line.strip() and len(line) < 2000:
                print(line.rstrip(), flush=True)
        result.wait()
    if result.returncode == 65:
        if not echo:
            print(log.read_text(encoding='utf-8').strip().splitlines()[-1])
        return False
    if result.returncode:
        if not echo:
            print(log.read_text(encoding='utf-8')[-7000:])
        raise RuntimeError(f'{stage} failed with exit {result.returncode}. Read the diagnostics above; '
                           f'the full log is {log}.')
    return True

def show_json(relative):
    value = json.loads((ROOT / relative).read_text())
    print(json.dumps(value, indent=2))
    return value

def show_previews(pattern):
    from IPython.display import Image, display
    for path in sorted((ROOT / 'outputs').glob(pattern))[:6]:
        print(path.name)
        display(Image(filename=str(path)))

def show_fit_diagnostics(protocol, role, rows):
    # Display only: values the worker already saved for every scored image (scored pixels n_valid and, for the
    # reference-assisted alignment, the fit's sign, rank and far-cap fraction).
    from IPython.display import Markdown, display
    table = [f'*Scoring support and fit diagnostics — {protocol}, {role}*', '',
             '| Model | Images | Fewest scored pixels (image) | Negative-scale fits | Degenerate fits | '
             'Far-cap fraction, mean / max |', '|---|---:|---|---:|---:|---|']
    flagged = []
    for model in sorted({r['model'] for r in rows}):
        group = [r for r in rows if r['model'] == model]
        fewest = min(group, key=lambda r: r['n_valid'])
        fitted = [r for r in group if 'rank' in r]
        if fitted:
            caps = [r['far_cap_fraction'] for r in fitted]
            cells = (f"{sum(r['negative_scale'] for r in fitted)} | {sum(r['degenerate'] for r in fitted)} | "
                     f"{sum(caps) / len(caps):.4f} / {max(caps):.4f}")
            marks = [(r['id'], ''.join(m for m, on in (('N', r['negative_scale']), ('D', r['degenerate']),
                                                       ('C', r['far_cap_fraction'] > 0)) if on)) for r in fitted]
            marked = [f'{i} [{m}]' for i, m in marks if m]
            if marked:
                flagged.append(f"{model}: {', '.join(marked)}")
        else:
            cells = 'no fit | no fit | no fit'
        table.append(f"| {model} | {len(group)} | {fewest['n_valid']} ({fewest['id']}) | {cells} |")
    display(Markdown('\n'.join(table)))
    if any('rank' in r for r in rows):
        print('Flagged images (N = negative scale, D = degenerate fit, C = far cap applied):',
              '; '.join(flagged) if flagged else 'none')

def show_metric_tables(filenames, role='test', breakdown=False):
    from IPython.display import Markdown, display
    rows = []
    for filename in filenames:
        rows.extend(json.loads((ROOT / 'outputs' / filename).read_text())['rows'])
    rows = [r for r in rows if r['role'] == role]
    for protocol in sorted({r['protocol'] for r in rows}):
        selected = [r for r in rows if r['protocol'] == protocol]
        for grouping in (['all', 'domain', 'scene'] if breakdown else ['all']):
            groups = {}
            for row in selected:
                key = (row['model'], row.get(grouping, 'all') if grouping != 'all' else 'all')
                groups.setdefault(key, []).append(row)
            table = [f'**{protocol} — {role}; equal-image means ({grouping})**', '',
                     '| Model | Group | Images | AbsRel ↓ | Delta-1 ↑ |', '|---|---|---:|---:|---:|']
            for (model, group), records in sorted(groups.items()):
                a = sum(r['abs_rel'] for r in records) / len(records)
                d = sum(r['delta1'] for r in records) / len(records)
                table.append(f'| {model} | {group} | {len(records)} | {a:.5f} | {d:.5f} |')
            display(Markdown('\n'.join(table)))
        show_fit_diagnostics(protocol, role, selected)

def show_activity():
    from IPython.display import Markdown, display
    activity = json.loads((ROOT / 'outputs/brightness_experiment.json').read_text())
    table = ['| Brightness factor | Improved AbsRel | Unchanged | Worsened | Mean paired AbsRel delta |',
             '|---|---:|---:|---:|---:|']
    for factor, comparison in sorted(activity['comparisons'].items()):
        delta = sum(r['abs_rel_delta'] for r in comparison['rows']) / len(comparison['rows'])
        table.append(f"| {factor} | {comparison['improved']} | {comparison['unchanged']} | {comparison['worsened']} | {delta:+.5f} |")
    display(Markdown('\n'.join(table)))
    scene_table = ['| Factor | Scene | Images | Mean paired AbsRel delta |', '|---|---|---:|---:|']
    for factor, comparison in sorted(activity['comparisons'].items()):
        for scene, values in sorted(comparison['per_scene'].items()):
            scene_table.append(f"| {factor} | {scene} | {values['n']} | {values['mean_abs_rel_delta']:+.5f} |")
    display(Markdown('\n'.join(scene_table)))
    print('Role:', activity['role'], '| Selection changed:', activity['selection_changed'])
    print('Full paired rows, clipping and input hashes are in brightness_experiment.json.')