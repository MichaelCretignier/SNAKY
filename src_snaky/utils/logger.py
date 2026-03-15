import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.text import Text
from rich.highlighter import ReprHighlighter, RegexHighlighter

# Define your theme
theme = Theme({
    'logging.level.debug':    'cyan',
    'logging.level.info':     'green',
    'logging.level.warning':  'yellow',
    'logging.level.error':    'red',
    'logging.level.critical': 'bold red',
})

class SnakyHandler(RichHandler):
    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        return Text(f'[{level_name}]', style=f'logging.level.{level_name.lower()}')
    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        text = super().render_message(record, message)
        if (record.levelname.lower() != 'info'):
            text.stylize(f'logging.level.{record.levelname.lower()}')
        return text

class SnakyHighlighter(RegexHighlighter):
    base_style = 'repr.'
    highlights = [
        *ReprHighlighter.highlights,
        r'(?P<path>[\w\/\-\.]+\.fits)',
        r'(?P<path>[\w\/\-\.]+\.p)',
    ]

# region Levels
INFO_BEGIN_LEVEL_NUM = 9
logging.addLevelName(INFO_BEGIN_LEVEL_NUM, "BEGIN")
# endregion

def setup(level: int = logging.INFO) -> logging.Logger:
    console = Console(theme=theme)

    handler = SnakyHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=False,
        show_level=True,
        show_path=False,
        log_time_format='%H:%M:%S',
        markup=True,
        highlighter=SnakyHighlighter(),
    )

    logger = logging.getLogger('snaky')
    logger.setLevel(level)
    logger.propagate = False
    logger.addHandler(handler)
    return logger
