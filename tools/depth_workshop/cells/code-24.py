RUN_UNLABELLED = False  # @param {type:"boolean"}
IMAGE_PATH = ''  # @param {type:"string"}
if RUN_UNLABELLED and not IMAGE_PATH.strip():
    print('Refused: IMAGE_PATH is empty. Set it to an RGB image within the documented size limits,'
          ' and run this cell again.')
elif RUN_UNLABELLED:
    if run_stage('infer', '--image', str(Path(IMAGE_PATH).resolve())):
        inference_root = Path(json.loads((ROOT / 'unlabelled_output.json').read_text())['root'])
        print((inference_root / 'result.json').read_text())
        from IPython.display import Image as ShowImage, display
        display(ShowImage(filename=str(inference_root / 'preview.png')))
        print('Files (download from the Files pane):', *sorted(str(p) for p in inference_root.iterdir()), sep='\n  ')
else:
    print('Optional unlabelled inference disabled.')