

def state_to_id(r: int, c: int, n_cols: int) -> int:
    """
    Convert (row, col) into a single integer id in order to flatten the dimension for the q table

    """
    return r * n_cols + c

def id_to_state(s: int, n_cols: int) -> tuple[int, int]:
    """
    Convert integer id back to (row, col).
    """
    r = s // n_cols
    c = s % n_cols
    return r, c