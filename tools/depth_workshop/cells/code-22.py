RUN_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}
if RUN_BYOD and not BYOD_PATH.strip():
    print('Refused: BYOD_PATH is empty. Set it to a directory containing manifest.json and labelled RGB-D files,'
          ' and run this cell again.')
elif RUN_BYOD:
    if run_stage('byod', '--byod', str(Path(BYOD_PATH).resolve())):
        byod_root = Path(json.loads((ROOT / 'byod_output.json').read_text())['root'])
        from IPython.display import Markdown, display
        display(Markdown((byod_root / 'outputs/summary.md').read_text()))
        print('Your results ZIP:', byod_root / 'outputs/depth_results.zip')
        print('Download it from the Files pane (right-click > Download). Review it before sharing: '
              'record IDs, labels and metrics remain inside it.')
else:
    print('Optional labelled BYOD disabled.')