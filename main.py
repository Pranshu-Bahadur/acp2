from pathlib import Path
def get_dir_files(dpath : str, fext : str = ".py") -> str:
    return list(Path(dpath).glob(f"**/*{fext}"))
repo = get_dir_files('/content/acp2')

for f in repo:
  exec(open(f).read())

CONFIG = eval(open('/content/drive/MyDrive/acp/config.json').read())


