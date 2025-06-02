import json

from importlib_resources import files
from langchain_core.prompts import PromptTemplate

json_content = files('resources').joinpath("templates.json").read_text(encoding='utf-8')
TEMPLATES = json.loads(json_content)


def template_to_content(template_name: str, fields: dict[str, str] = None) -> str:
    """
    Convert a template  to an argument dictionary.
    """
    if not fields:
        fields = TEMPLATES[template_name]['default']

    response = PromptTemplate.from_template(TEMPLATES[template_name]['template']).invoke(fields)
    return response.text
