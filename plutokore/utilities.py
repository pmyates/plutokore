from contextlib import contextmanager as _contextmanager
import os as _os
import sys as _sys

def video(fname, mimetype):  #pragma: no cover
    """Displays a video in an jupyter notebook"""
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video autoplay controls alt="test" src="data:video/{0};base64,{1}">'.format(
        mimetype, video_encoded)
    return HTML(data=video_tag)


@_contextmanager
def suppress_stdout():
    """Suppresses stdout"""
    from contextlib import contextmanager as _contextmanager
    with open(_os.devnull, "w") as devnull:
        old_stdout = _sys.stdout
        _sys.stdout = devnull
        try:
            yield
        finally:
            _sys.stdout = old_stdout

def printmd(string):
    from IPython.display import display as _display
    from IPython.display import Markdown as _Markdown
    """Displays a markdown string in the jupyter notebook"""
    _display(_Markdown(string))

def close_open_hdf5_files():  #pragma: no cover
    """Close any open hdf5 handles"""
    import gc
    import h5py
    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, h5py.File):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass  # Was already closed
class tcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    LIGHT_CYAN = '\033[96m'
    LIGHT_YELLOW = '\033[93m'
    LIGHT_RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def print_with_color(cls, colour, s, **kwargs):
        print(colour + s + cls.ENDC, **kwargs)
