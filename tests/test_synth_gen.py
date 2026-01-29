import os

from scripts.gen_synth_data import generate_synth_data
from src.indexer import build_index, read_roots_txt


def test_synth_gen_filenames_and_group_consistency(tmp_path):
    root = tmp_path / "synthetic_root"
    generate_synth_data(root=str(root), s_max=3, runs=2)

    roots_txt = root / "roots.txt"
    roots = read_roots_txt(str(roots_txt))
    assert len(roots) == 2

    files = sorted([f for f in os.listdir(roots[0]) if f.endswith(".jpg")])
    assert len(files) == 2 * 2 * 3
    assert "1_10X1001.jpg" in files
    assert "2_10X2003.jpg" in files

    entries, stats = build_index(roots)
    assert stats[(roots[0], 1, 1)].w_block == 1228
    assert stats[(roots[0], 1, 2)].w_block == 819
    assert stats[(roots[1], 2, 1)].w_block == 1228
    assert stats[(roots[1], 2, 2)].w_block == 819
