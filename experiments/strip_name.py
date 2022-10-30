from pathlib import Path


def strip_name_in_file(exp_path: Path):
    text = exp_path.read_text()
    text = text.replace("/home/shiayupov", '~')
    exp_path.write_text(text)


if __name__ == "__main__":
    cur_dir = Path(__file__).absolute().parent
    for filepath in cur_dir.iterdir():
        if filepath.name in ['README.md', 'strip_name.py']:
            continue
        print(f"stripping {filepath.name}")
        strip_name_in_file(filepath)
    print("DONE")
