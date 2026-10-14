from pyrle import Rle, RleDict


def _rle():
    return Rle([1, 2], [0, 1])


def test_two_character_chromosome_is_unstranded():
    rd = RleDict({"2L": _rle(), "2R": _rle()})
    assert not rd.stranded
    assert not rd.__dict__["stranded"]
    s = str(rd)
    assert "Unstranded RleDict" in s
    assert "2L\n--\n" in s


def test_non_string_keys_render():
    unstranded = RleDict({1: _rle(), 2: _rle(), 3: _rle()})
    assert not unstranded.stranded
    assert "Unstranded RleDict object with 3 chromosomes." in str(unstranded)

    stranded = RleDict({(1, "+"): _rle(), (2, "-"): _rle()})
    assert stranded.stranded
    s = str(stranded)
    assert "1 +\n---\n" in s
    assert "2 -" in s


def test_empty_dict_uses_stranded_argument():
    assert RleDict({}, stranded=True).__dict__["stranded"]
    assert str(RleDict({})) == "Empty RleDict."
