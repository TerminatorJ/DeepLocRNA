# %%
import click

# %%
from .bin import train

# %%
@click.group()
def main():
    pass

# %%
main.add_command(train.main, name='train')

# %%
if __name__ == '__main__':
    main()