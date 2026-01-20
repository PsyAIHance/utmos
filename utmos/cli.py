from pathlib import Path

import click

from .score import Score

@click.command()
@click.argument('path', type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option(
    '--recursive/--no-recursive',
    default=True,
    help='Search folders recursively.',
)
def main(path, recursive):
    model = Score()
    target = Path(path)
    if target.is_dir():
        score = model.calculate_wav_files(target, recursive=recursive)
        print(f"Average Score: {score}")
    else:
        score = model.calculate_wav_file(target)
        print(f"Score: {score}")

if __name__ == '__main__':
    main()
