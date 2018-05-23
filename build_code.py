import sys

from jinja2 import Environment, FileSystemLoader
import yaml

j2_env = Environment(loader=FileSystemLoader("."),
                    trim_blocks=True)

d_add_sub = yaml.load(open("template_filler_add_sub.yaml"))
d_div = yaml.load(open("template_filler_div.yaml"))

template = "rle_arithmetic_template.j2"

code = j2_env.get_template(template).render(
    configs_add_sub=d_add_sub.values(),
    configs_div=d_div.values(),
)

open("pyrle/src/rle.pyx", "w+").write(code)
