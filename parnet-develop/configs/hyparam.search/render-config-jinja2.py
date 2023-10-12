# %%
import argparse
from jinja2 import Environment, FileSystemLoader, select_autoescape

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='<config.jinja2>')
    args = parser.parse_args()

    env = Environment(loader=FileSystemLoader('./'), autoescape=select_autoescape())
    template = env.get_template(args.config)
    print(template.render())    

# %%
if __name__ == '__main__':
    main()