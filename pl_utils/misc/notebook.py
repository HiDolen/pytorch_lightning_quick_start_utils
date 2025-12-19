def is_notebook() -> bool:
    """
    判断当前环境是否为 Jupyter Notebook。
    """
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell != None
    except NameError:
        return False
