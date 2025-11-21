import string


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        else:
            return super().get_value(key, args, kwargs)

    def parse(self, format_string):
        try:
            return super().parse(format_string)
        except ValueError:
            return [(format_string, None, None, None)]


_formatter = _SafeFormatter()

def safe_format(format_string, *args, **kwargs):
    return _formatter.format(format_string, *args, **kwargs)
