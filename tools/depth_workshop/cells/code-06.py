print('Reading the 120 pinned DIODE files (about 312 MB) from the ETH Zurich archive. This took about'
      ' 10 minutes in a recorded Colab T4 run and prints nothing until it finishes.', flush=True)
run_stage('prepare')
show_json('outputs/prepare.json')
show_previews('reference*.png')