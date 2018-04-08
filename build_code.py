import sys

from jinja2 import Environment, FileSystemLoader
import yaml

j2_env = Environment(loader=FileSystemLoader("."),
                    trim_blocks=True)

d = yaml.load(open("template_filler.yaml"))
template = "rle_arithmetic_template.j2"

code = j2_env.get_template(template).render(
    configs=d.values()
)

open("pyrle.pyx", "w+").write(code)
